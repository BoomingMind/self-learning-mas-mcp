"""
src/agents/progress_coach.py

The Progress Coach agent with A2A delegation support.

Reads quiz results, generates personalized coaching messages,
updates topic status in the roadmap, and optionally delegates
to the external Quiz A2A service or the CrewAI Study Buddy
for supplementary help. Falls back gracefully when external
services are unavailable.
"""

import json
import os
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from graph.state import QuizResult, StudyRoadmap, get_latest_quiz_result
from mcp_servers.memory_server import memory_set


MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PASS_THRESHOLD = 0.5

# A2A service URL, read from env so it's configurable
QUIZ_SERVICE_URL = os.getenv("QUIZ_SERVICE_URL", "http://localhost:9001")


COACHING_PROMPT = """You are an encouraging learning coach reviewing a student's quiz results.

Provide a brief, warm coaching message (2-3 sentences max) based on:
  - The topic studied
  - Their score (0.0 = 0%, 1.0 = 100%)
  - Any weak areas identified

Return ONLY valid JSON:
{{
  "summary": "2-3 sentence encouraging summary",
  "encouragement": "One short motivational sentence for next steps"
}}

Be specific, reference the topic and any weak areas by name.
Never be discouraging. A low score means "more practice needed", not "you failed."
"""


def get_coaching_message(topic: str, score: float, weak_areas: list[str]) -> dict:
    """Ask the LLM for a personalised coaching message."""
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.4,
        format="json",
    )

    context = {
        "topic":         topic,
        "score_percent": f"{score:.0%}",
        "weak_areas":    weak_areas if weak_areas else ["none identified"],
    }

    try:
        response = llm.invoke([
            SystemMessage(content=COACHING_PROMPT),
            HumanMessage(content=json.dumps(context)),
        ])
    except Exception as e:
        print(f"[Progress Coach] LLM call failed: {e}")
        return {
            "summary": f"You scored {score:.0%} on {topic}. Keep going!",
            "encouragement": "Every topic builds on the last.",
        }

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {
            "summary":      f"You scored {score:.0%} on {topic}.",
            "encouragement": "Keep going, every topic builds on the last!",
        }


def try_a2a_quiz_delegation(
    topic: str,
    explanation: str,
    answers: list[str],
) -> dict | None:
    """
    Attempt to delegate quiz grading to the A2A Quiz Service.

    Returns the grading result dict if successful, None if the
    service is unavailable or returns an error.

    The Progress Coach calls this first. If it returns None,
    the coach falls back to local quiz generation.
    """
    # Read at call time, not module load time (fixes test timing bug)
    use_a2a = os.getenv("USE_A2A_QUIZ", "true").lower() == "true"
    if not use_a2a:
        return None

    try:
        from a2a_services.a2a_client import delegate_quiz_task, is_quiz_service_available

        if not is_quiz_service_available(QUIZ_SERVICE_URL):
            print("[Progress Coach] Quiz A2A service not available at "
                  f"{QUIZ_SERVICE_URL}, using local quiz generator")
            return None

        print(f"[Progress Coach] Delegating quiz to A2A service: {QUIZ_SERVICE_URL}")
        result = delegate_quiz_task(
            topic=topic,
            explanation=explanation,
            answers=answers,
            quiz_service_url=QUIZ_SERVICE_URL,
        )

        if "error" in result:
            print(f"[Progress Coach] A2A delegation failed: {result['error']}")
            return None

        print(f"[Progress Coach] A2A quiz complete: score={result.get('score', 0):.0%}")
        return result

    except ImportError:
        return None
    except Exception as e:
        print(f"[Progress Coach] A2A error: {e}")
        return None


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

        if "error" in result or result.get("status") == "error":
            return None

        return result.get("assistance", "")

    except Exception as e:
        print(f"[Progress Coach] Study Buddy error: {e}")
        return None


def progress_coach_node(state: dict) -> dict:
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
    coaching = get_coaching_message(latest.topic, score, latest.weak_areas)

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
    _topic_obj = topics[idx] if idx < len(topics) else None
    _status = (
        "done" if _topic_obj is None
        else _topic_obj.get("status", "done") if isinstance(_topic_obj, dict)
        else _topic_obj.status
    )
    progress_data = json.dumps({
        "topic":      latest.topic,
        "score":      score,
        "weak_areas": latest.weak_areas,
        "status":     _status,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    })
    memory_set(session_id, f"progress_topic_{idx}", progress_data)

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
    if score < PASS_THRESHOLD and latest.weak_areas:
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
            weak_areas=latest.weak_areas,
        )

        if assistance:
            print(f"\n{'─'*60}")
            print("Study Buddy (via CrewAI → A2A):")
            print(assistance)
            print(f"{'─'*60}\n")

    return {
        "roadmap":               roadmap,
        "current_topic_index":   next_idx,
        "messages":              [AIMessage(content=coaching["summary"])],
        "error":                 None,
    }
