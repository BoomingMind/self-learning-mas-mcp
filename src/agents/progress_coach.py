import json
import os
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import QuizResult, StudyRoadmap, get_latest_quiz_result
from mcp_servers.memory_server import memory_set

MODEL_NAME = os.getenv("OPEN_OPENROUTER_MODEL", "thinkingmachines/inkling:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
PASS_THRESHOLD = 0.5

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

Be specific. Reference the topic and any weak areas by name.
Never be discouraging. A low score means "more practice needed", not "you failed."
"""

def get_coaching_message(topic: str, score: float, weak_areas: list[str]) -> dict:
    """Ask the LLM for a personalised coaching message."""
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        temperature=0.3,
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
        return json.loads(response.content)
    except Exception as e:
        print(f"[Progress Coach] LLM call failed: {e}")
        return {
            "summary":      f"You scored {score:.0%} on {topic}. Keep going!",
            "encouragement": "Every topic builds on the last.",
        }

def progress_coach_node(state: dict) -> dict:
    """
    LangGraph node: Progress Coach

    Reads:  state["quiz_results"], state["roadmap"],
            state["current_topic_index"], state["session_id"]
    Writes: state["roadmap"], state["current_topic_index"],
            state["messages"], state["error"]
    """
    latest = get_latest_quiz_result(state)
    if latest is None:
        return {"error": "No quiz results. Quiz Generator must run first"}

    roadmap = state.get("roadmap")
    if roadmap is None:
        return {"error": "No roadmap found"}

    idx = state.get("current_topic_index", 0)
    session_id = state.get("session_id", "unknown")
    score = latest.score

    print(f"\n[Progress Coach] Topic: '{latest.topic}'")
    print(f"[Progress Coach] Score: {score:.0%}")
    if latest.weak_areas:
        print(f"[Progress Coach] Weak areas: {', '.join(latest.weak_areas)}")

    # Get coaching message from LLM
    coaching = get_coaching_message(latest.topic, score, latest.weak_areas)

    # Update topic status in the roadmap
    topics = roadmap.get("topics", []) if isinstance(roadmap, dict) else roadmap.topics
    if idx < len(topics):
        topic = topics[idx]
        new_status = "completed" if score >= PASS_THRESHOLD else "needs_review"
        if isinstance(topic, dict):
            topic["status"] = new_status
        else:
            topic.status = new_status

    # Advance the topic index
    next_idx = idx + 1
    all_done = next_idx >= len(topics)

    # Persist progress to MCP memory
    memory_set(session_id, f"progress_topic_{idx}", json.dumps({
        "topic":      latest.topic,
        "score":      score,
        "weak_areas": latest.weak_areas,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }))

    # Print coaching feedback
    print(f"\n{'─'*60}")
    print(f"Coach: {coaching['summary']}")
    print(f"{coaching['encouragement']}")

    if all_done:
        results = state.get("quiz_results", [])
        avg = sum(r.score for r in results) / max(len(results), 1)
        print(f"\nSession complete! Average: {avg:.0%}")
    else:
        next_topic = topics[next_idx]
        next_title = next_topic.get("title") if isinstance(next_topic, dict) else next_topic.title
        print(f"\nNext topic: '{next_title}'")
    print(f"{'─'*60}\n")

    return {
        "roadmap":              roadmap,
        "current_topic_index":  next_idx,
        "messages":             [AIMessage(content=coaching["summary"])],
        "error":                None,
    }