"""
src/a2a_services/a2a_client.py

Client utilities for calling A2A services.

The Progress Coach uses this to delegate quiz tasks to the
Quiz Generator A2A service instead of calling it directly.

Why a separate client module?
  Keeps the HTTP/protocol details out of agent code.
  The Progress Coach just calls delegate_quiz_task() and gets
  a result dict back, it doesn't need to know anything about
  JSON-RPC, Agent Cards, or HTTP.
"""

import json
import os
import uuid
import httpx

# Read from env so non-localhost deployments work correctly.
# These are the defaults used when callers don't pass an explicit URL.
QUIZ_SERVICE_URL = os.getenv("QUIZ_SERVICE_URL", "http://localhost:9001")

# How long to wait for the quiz service to respond.
# Quiz generation + grading takes 15-60s depending on model size.
DEFAULT_TIMEOUT = 120.0


def discover_agent(base_url: str) -> dict:
    """
    Fetch an agent's Agent Card to discover its capabilities.

    Args:
        base_url: The agent's base URL (e.g. 'http://localhost:9001')

    Returns:
        The parsed Agent Card dict, or {} if unreachable.
    """
    card_url = f"{base_url.rstrip('/')}/.well-known/agent-card.json"
    try:
        response = httpx.get(card_url, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[A2A Client] Cannot reach {card_url}: {e}")
        return {}


def send_task(
    base_url: str,
    message_text: str,
    task_id: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """
    Submit a task to an A2A agent and return the result.

    Constructs a JSON-RPC 2.0 tasks/send request, sends it,
    and extracts the result text from the response envelope.

    Args:
        base_url:     Agent base URL.
        message_text: JSON string payload for the task.
        task_id:      Optional task ID (auto-generated if not provided).
        timeout:      Seconds to wait before giving up.

    Returns:
        Parsed result dict from the agent, or {"error": ...} on failure.
    """
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "tasks/send",
        "params": {
            "id":      task_id or str(uuid.uuid4()),
            "message": {
                "role":  "user",
                "parts": [{"type": "text", "text": message_text}],
            },
        },
    }

    url = f"{base_url.rstrip('/')}/tasks/send"
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        # Extract text from the A2A response envelope
        # Structure: result.artifacts[0].parts[0].text
        result = data.get("result", {})
        artifacts = result.get("artifacts", [])
        if artifacts:
            for part in artifacts[0].get("parts", []):
                if part.get("type") == "text":
                    try:
                        return json.loads(part["text"])
                    except json.JSONDecodeError:
                        return {"text": part["text"]}

        # Fallback: check status message
        status = result.get("status", {})
        if status:
            msg = status.get("message", {})
            for part in msg.get("parts", []):
                if part.get("type") == "text":
                    try:
                        return json.loads(part["text"])
                    except json.JSONDecodeError:
                        return {"text": part["text"]}

        return result

    except httpx.TimeoutException:
        return {"error": f"Quiz service timed out after {timeout}s"}
    except httpx.ConnectError:
        return {"error": f"Cannot connect to quiz service at {url}"}
    except Exception as e:
        return {"error": f"A2A task failed: {type(e).__name__}: {e}"}


def delegate_quiz_task(
    topic: str,
    explanation: str,
    answers: list[str] | None = None,
    quiz_service_url: str = QUIZ_SERVICE_URL,
) -> dict:
    """
    High-level helper: delegate a quiz task to the Quiz A2A service.

    This is the function the Progress Coach calls. It handles all the
    A2A protocol details, JSON-RPC envelope, response parsing, errors.

    Args:
        topic:            Topic to quiz on.
        explanation:      The Explainer's output (context for question gen).
        answers:          Optional list of pre-collected answers.
                          If None or empty, service returns questions only.
        quiz_service_url: URL of the Quiz A2A service.

    Returns:
        Result dict with keys:
          status:   "questions_ready" | "graded" | "error"
          topic:    the topic
          score:    average score (only when graded)
          weak_areas: list of missed concepts (only when graded)
          questions: list of question dicts (always present on success)
    """
    payload = json.dumps({
        "topic":       topic,
        "explanation": explanation,
        "answers":     answers or [],
    })

    return send_task(quiz_service_url, payload)


def is_quiz_service_available(
    quiz_service_url: str = QUIZ_SERVICE_URL,
) -> bool:
    """
    Quick health check: is the quiz service reachable?

    Used by the Progress Coach to decide whether to use A2A or
    fall back to the local quiz generator.

    Returns True if the Agent Card can be fetched, False otherwise.
    """
    card = discover_agent(quiz_service_url)
    return bool(card)


STUDY_BUDDY_URL = os.getenv("STUDY_BUDDY_URL", "http://localhost:9002")


def request_study_assistance(
    topic: str,
    explanation: str,
    weak_areas: list[str] | None = None,
    study_buddy_url: str = STUDY_BUDDY_URL,
) -> dict:
    """
    Request supplementary study assistance from the CrewAI Study Buddy.

    Called by the Progress Coach when a student scores below the pass
    threshold and could benefit from a different explanation angle.

    Args:
        topic:           The topic the student is studying.
        explanation:     The original Explainer output (context).
        weak_areas:      Concepts the student struggled with.
        study_buddy_url: URL of the CrewAI Study Buddy A2A service.

    Returns:
        Result dict with keys:
          source:     "crewai_study_buddy"
          topic:      the topic
          assistance: supplementary explanation text
          status:     "complete" | "error"
    """
    payload = json.dumps({
        "topic":       topic,
        "explanation": explanation,
        "weak_areas":  weak_areas or [],
    })

    return send_task(study_buddy_url, payload, timeout=180.0)


def is_study_buddy_available(
    study_buddy_url: str = STUDY_BUDDY_URL,
) -> bool:
    """Check if the CrewAI Study Buddy service is reachable."""
    card = discover_agent(study_buddy_url)
    return bool(card)
