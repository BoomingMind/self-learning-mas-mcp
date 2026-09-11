"""
src/a2a_services/a2a_client.py

Client utilities for calling A2A services.

The Progress Coach uses this to request supplementary help from the
CrewAI Study Buddy A2A service.

Why a separate client module?
  Keeps the HTTP/protocol details out of agent code.
  The Progress Coach calls the Study Buddy helper and gets a result dict back;
  it doesn't need to know anything about
  JSON-RPC, Agent Cards, or HTTP.
"""

import json
import os
import uuid
import httpx

# How long to wait for an A2A agent to respond.
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

    Constructs a JSON-RPC 2.0 message/send request, sends it,
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
        "method":  "message/send",
        "params": {
            "message": {
                "messageId": task_id or str(uuid.uuid4()),
                "kind": "message",
                "role":  "user",
                "parts": [{"type": "text", "text": message_text}],
            },
        },
    }

    url = base_url.rstrip("/")
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

        # Current A2A message/send responses return a Message directly.
        for part in result.get("parts", []):
            if part.get("kind") == "text" or part.get("type") == "text":
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
        return {"error": f"A2A service timed out after {timeout}s"}
    except httpx.ConnectError:
        return {"error": f"Cannot connect to A2A service at {url}"}
    except Exception as e:
        return {"error": f"A2A task failed: {type(e).__name__}: {e}"}


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
