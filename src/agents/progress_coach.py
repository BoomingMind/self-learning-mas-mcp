"""
src/agents/progress_coach.py

The Progress Coach agent with optional Study Buddy A2A support.

Reads quiz results, generates personalized coaching messages,
updates topic status in the roadmap, and optionally delegates
to the CrewAI Study Buddy for supplementary help. Quiz generation and grading
are completed by the preceding native LangGraph Quiz Generator node.
"""

import json
import os
import asyncio
from datetime import datetime, timezone

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from graph.state import get_latest_quiz_result
from mcp_client import call_tool
from model_config import build_chat_model


PASS_THRESHOLD = 0.5

COACHING_PROMPT = """You are an encouraging coaching agent reviewing a student's quiz results.

Use the topic, score, and weak areas in the user request to provide a brief,
warm coaching message (2-3 sentences max).

Your final response must be ONLY valid JSON:
{{
  "summary": "2-3 sentence encouraging summary",
  "encouragement": "One short motivational sentence for next steps",
  "recommendation": "Specific recommendation for what the learner should do next",
  "review_focus": ["specific concepts to review"]
}}

Be specific, reference the topic and any weak areas by name.
Never be discouraging. A low score means "more practice needed", not "you failed."
"""


def _content_text(content: object) -> str:
    """Normalize text/content blocks returned by different chat model providers."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, str) or isinstance(block, dict)
        )
    return str(content)


def _coaching_fallback(topic: str, score: float, weak_areas: list[str]) -> dict:
    return {
        "summary": f"You scored {score:.0%} on {topic}.",
        "encouragement": "Keep going, every topic builds on the last!",
        "recommendation": (
            f"Review {', '.join(weak_areas)} before trying another quiz."
            if weak_areas else "Continue practicing with another example."
        ),
        "review_focus": weak_areas,
    }


def get_coaching_message(
    topic: str,
    score: float,
    weak_areas: list[str],
    model_provider: str = "ollama",
    model_name: str = "",
    callbacks: list | None = None,
) -> dict:
    """Ask the LLM for a personalised coaching message."""
    llm = build_chat_model(
        provider=model_provider, model=model_name or None,
        temperature=0.4,
        json_mode=True,
        reasoning_effort="low",
    )

    context = {
        "topic":         topic,
        "score_percent": f"{score:.0%}",
        "weak_areas":    weak_areas if weak_areas else ["none identified"],
    }

    try:
        request = {"messages": [HumanMessage(content=json.dumps(context))]}
        agent = create_agent(
            model=llm,
            system_prompt=COACHING_PROMPT,
            name="progress_coach",
        )
        invoke_config = {"callbacks": callbacks} if callbacks else None
        result = (
            agent.invoke(request, config=invoke_config)
            if invoke_config else agent.invoke(request)
        )
        response = result["messages"][-1]
        print(f"[Progress Coach] Input:\n{json.dumps(context)}")
        print(f"[Progress Coach] Output:\n{response.content}")
    except Exception as e:
        print(f"[Progress Coach] LLM call failed: {e}")
        return _coaching_fallback(topic, score, weak_areas)

    try:
        text = _content_text(response.content).strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("coaching response must be a JSON object")
        fallback = _coaching_fallback(topic, score, weak_areas)
        return {**fallback, **parsed}
    except json.JSONDecodeError:
        return _coaching_fallback(topic, score, weak_areas)
    except (TypeError, ValueError):
        return _coaching_fallback(topic, score, weak_areas)


def try_study_buddy_assistance(
    topic: str,
    explanation: str,
    weak_areas: list[str],
) -> str | None:
    """
    Request supplementary study help from the CrewAI Study Buddy.

    Called when a student scores below 0.5 and could benefit from
    a different explanation angle.

    Returns the assistance text if available, None if unavailable.
    The Progress Coach prints this to the user as bonus help.
    """
    study_buddy_url = os.getenv("STUDY_BUDDY_URL", "http://localhost:9002")
    use_study_buddy = os.getenv("USE_STUDY_BUDDY", "true").lower() == "true"

    if not use_study_buddy:
        return None

    try:
        from a2a_services.a2a_client import (
            request_study_assistance,
            is_study_buddy_available,
        )

        if not is_study_buddy_available(study_buddy_url):
            return None

        print("[Progress Coach] Requesting study assistance from CrewAI Study Buddy...")
        result = request_study_assistance(
            topic=topic,
            explanation=explanation,
            weak_areas=weak_areas,
            study_buddy_url=study_buddy_url,
        )

        if isinstance(result, str):
            return result
        if isinstance(result.get("text"), str):
            try:
                result = json.loads(result["text"])
            except json.JSONDecodeError:
                return result["text"]
        if "error" in result or result.get("status") == "error":
            return None

        assistance = result.get("assistance", "")
        return assistance if isinstance(assistance, str) else _content_text(assistance)

    except Exception as e:
        print(f"[Progress Coach] Study Buddy error: {e}")
        return None


def progress_coach_node(
    state: dict, config: RunnableConfig | None = None
) -> dict:
    """
    LangGraph node: Progress Coach

    Reads:
        state["quiz_results"]        : latest quiz result
        state["roadmap"]             : to update topic status
        state["current_topic_index"]: which topic we just finished
        state["session_id"]          : for MCP memory persistence

    Writes:
        state["roadmap"]             : topic status updated
        state["current_topic_index"]: incremented
        state["messages"]            : coaching message
        state["error"]               : error string on failure
    """
    latest = get_latest_quiz_result(state)
    if latest is None:
        return {"error": "No quiz results, Quiz Generator must run first"}

    roadmap = state.get("roadmap")  # may be StudyRoadmap, dict, or None after resume
    if roadmap is None:
        return {"error": "No roadmap found"}

    idx = state.get("current_topic_index", 0)
    session_id = state.get("session_id", "unknown")
    score = latest.score

    print(f"\n[Progress Coach] Topic: '{latest.topic}'")
    print(f"[Progress Coach] Score: {score:.0%}")
    if latest.weak_areas:
        print(f"[Progress Coach] Weak areas: {', '.join(latest.weak_areas)}")

    # ── Get coaching message ──────────────────────────────────────────
    coaching = get_coaching_message(
        latest.topic, score, latest.weak_areas,
        state.get("model_provider", "ollama"), state.get("model_name", ""),
        callbacks=(config or {}).get("callbacks"),
    )

    # ── Update topic status ───────────────────────────────────────────
    topics = roadmap.get("topics", []) if isinstance(roadmap, dict) else roadmap.topics
    if idx < len(topics):
        topic = topics[idx]
        new_status = "completed" if score >= PASS_THRESHOLD else "needs_review"
        if isinstance(topic, dict):
            topic["status"] = new_status
        else:
            topic.status = new_status

    # ── Advance to next topic ─────────────────────────────────────────
    next_idx = idx + 1
    all_done = next_idx >= len(topics)

    # ── Persist progress via MCP memory ──────────────────────────────
    # Safe status read, guard idx before subscripting
    topic_obj = topics[idx] if idx < len(topics) else None
    status = (
        "done"
        if topic_obj is None
        else topic_obj.get("status", "done")
        if isinstance(topic_obj, dict)
        else topic_obj.status
    )
    progress_data = json.dumps({
        "topic":      latest.topic,
        "score":      score,
        "weak_areas": latest.weak_areas,
        "status":     status,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    })
    try:
        asyncio.run(call_tool(
            "memory", "memory_set",
            {"session_id": session_id, "key": f"progress_topic_{idx}", "value": progress_data},
        ))
    except Exception as error:
        print(f"[Progress Coach] Memory unavailable; continuing: {error}")

    # ── Print coaching message ────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Coach: {coaching['summary']}")
    print(f"{coaching['encouragement']}")

    if all_done:
        completed = sum(1 for t in topics if (t.get("status") if isinstance(t, dict) else t.status) == "completed")
        total = len(topics)
        results = state.get("quiz_results", [])
        avg = sum(r.score for r in results) / max(len(results), 1)
        print(f"\nSession complete! {completed}/{total} topics passed.")
        print(f"Overall average: {avg:.0%}")
    else:
        next_topic = topics[next_idx]
        next_title = next_topic.get("title") if isinstance(next_topic, dict) else next_topic.title
        print(f"\nNext topic: '{next_title}'")
    print(f"{'─'*60}\n")

    # ── Optional: CrewAI Study Buddy for low scores ───────────────────
    # When a student scores below the pass threshold, request supplementary
    # help from the CrewAI Study Buddy via A2A.
    # This is where LangGraph calls CrewAI through the A2A protocol.
    assistance = None
    if score < PASS_THRESHOLD:
        # Extract the most recent explanation from messages
        explanation = ""
        for msg in reversed(state.get("messages", [])):
            if (isinstance(msg, AIMessage) and msg.content
                    and not getattr(msg, "tool_calls", None)):
                explanation = msg.content
                break

        assistance = try_study_buddy_assistance(
            topic=latest.topic,
            explanation=explanation,
            weak_areas=latest.weak_areas or ["general understanding"],
        )

        if assistance:
            print(f"\n{'─'*60}")
            print("Study Buddy (via CrewAI → A2A):")
            print(assistance)
            print(f"{'─'*60}\n")

    return {
        "roadmap":               roadmap,
        "current_topic_index":   next_idx,
        "explainer_status": "CONTINUE",
        "explainer_iterations": 0,
        "quiz_requested": False,
        "awaiting_quiz_approval": False,
        "messages":              [AIMessage(content=coaching["summary"])],
        "coaching_summary":      coaching["summary"],
        "coaching_encouragement": coaching["encouragement"],
        "coaching_recommendation": coaching.get("recommendation", ""),
        "coaching_review_focus": coaching.get("review_focus", []),
        "study_buddy_assistance": assistance or "",
        "coaching_topic_index":  idx,
        "coaching_topic":        latest.topic,
        "error":                 None,
    }
