"""
src/agents/curriculum_planner.py

The Curriculum Planner agent.

Responsibility: take a learning goal string and produce a
structured StudyRoadmap with ordered topics, time estimates,
and prerequisites.

This agent makes one LLM call with structured JSON output
  - Deterministic parsing into a StudyRoadmap dataclass

It demonstrates the foundational pattern every agent follows:
  read from state → call LLM → parse output → return state update
"""

import json
import os
import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from graph.state import StudyRoadmap, Topic
from mcp_client import call_tool
from model_config import build_chat_model


# ─────────────────────────────────────────────────────────────────────────────
# Model configuration
#
# Read from .env so you can switch models without touching code.
# Defaults to qwen2.5:7b which works on 8GB VRAM.
# ─────────────────────────────────────────────────────────────────────────────

def should_research_goal(goal: str) -> bool:
    return any(term in goal.lower() for term in (
        "latest", "current", "2025", "2026", "framework", "library",
        "prerequisite", "resources", "roadmap",
    ))


def planner_research(goal: str) -> str:
    if not should_research_goal(goal):
        return ""
    try:
        result = asyncio.run(call_tool(
            "tavily", "search_web",
            {"query": f"{goal} official prerequisites learning resources", "max_results": 3},
        ))
        return f"\nUntrusted research context:\n{json.dumps(result)[:6000]}"
    except Exception as error:
        print(f"[Curriculum Planner] Web research unavailable: {error}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
#
# The system prompt is the "job description" for this agent.
# It stays constant across all calls, only the human message changes.
#
# Key decisions in this prompt:
#   1. "Return ONLY valid JSON", no prose, no markdown fences
#   2. Explicit schema with field names and types
#   3. Clear rules to constrain the output space
#   4. 4–6 topics, enough to demonstrate the study loop
#      without making sessions excessively long during development
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """You are an expert curriculum designer. Your job is to
create a structured study roadmap when given a learning goal.

Return ONLY valid JSON with no prose, no markdown code fences, no explanation.
The JSON must match this exact schema:

{
  "goal": "the original learning goal exactly as given",
  "total_weeks": <integer between 1 and 12>,
  "weekly_hours": <integer between 3 and 10>,
  "topics": [
    {
      "title": "Short topic name (3-6 words)",
      "description": "One clear sentence explaining what this topic covers",
      "estimated_minutes": <integer between 30 and 120>,
      "prerequisites": ["title of earlier topic if required, else empty list"],
      "status": "pending"
    }
  ]
}

Rules:
- Order topics from foundational to advanced
- prerequisites must reference earlier topic titles exactly as written
- estimated_minutes is time for one focused study session, not total time
- Aim for 4 to 6 topics, enough depth without being overwhelming
- Every topic must have a clear, specific description
- status must always be "pending"
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
#
# We create a new LLM instance per call rather than a module-level
# singleton. This avoids subtle state issues in long-running processes
# and makes it easier to test with different configurations.
# ─────────────────────────────────────────────────────────────────────────────

def build_planner_llm(provider: str = "ollama", model: str = ""):
    """
    Create the Ollama LLM client for the Curriculum Planner.

    temperature=0.1, very low temperature for structured output.
    Higher temperature introduces randomness that makes JSON parsing
    less reliable. For creative tasks (like explanations) we use 0.3.
    For structured output, stay at 0.1 or lower.

    format="json", enables Ollama's JSON mode. The model will never
    produce output that isn't valid JSON. This is a hard constraint
    at the inference level, not just a prompt instruction.
    """
    return build_chat_model(
        provider=provider, model=model or None,
        temperature=0.1,
        json_mode=True,
        reasoning_effort="low",
    )


def build_planner_agent(provider: str = "ollama", model: str = ""):
    """Build the LangChain agent used to create a roadmap."""
    return create_agent(
        model=build_planner_llm(provider, model),
        system_prompt=PLANNER_SYSTEM_PROMPT,
        name="curriculum_planner",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output parser
#
# Separated from the node function so it can be tested independently
# without running an LLM call.
# ─────────────────────────────────────────────────────────────────────────────

def parse_roadmap_json(json_string: str) -> StudyRoadmap:
    """
    Parse the LLM's JSON output into a StudyRoadmap dataclass.

    Args:
        json_string: Raw string output from the LLM.

    Returns:
        A populated StudyRoadmap instance.

    Raises:
        ValueError: If the JSON is invalid or missing required fields.
                    The node catches this and returns an error state.
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(
            "LLM returned invalid JSON.\n"
            f"Error: {e}\n"
            f"Raw output (first 300 chars): {json_string[:300]}"
        )

    # Validate required top-level fields
    required = ["goal", "total_weeks", "topics"]
    for field in required:
        if field not in data:
            raise ValueError(f"LLM JSON missing required field: '{field}'")

    if not isinstance(data["topics"], list) or len(data["topics"]) == 0:
        raise ValueError("LLM JSON 'topics' must be a non-empty list")

    # Build Topic objects
    topics = []
    for i, t in enumerate(data["topics"]):
        # Validate each topic has required fields
        for field in ["title", "description", "estimated_minutes"]:
            if field not in t:
                raise ValueError(
                    f"Topic {i} missing required field: '{field}'"
                )
        topics.append(Topic(
            title=t["title"],
            description=t["description"],
            estimated_minutes=int(t["estimated_minutes"]),
            prerequisites=t.get("prerequisites", []),
            status=t.get("status", "pending"),
        ))

    return StudyRoadmap(
        goal=data["goal"],
        total_weeks=int(data["total_weeks"]),
        weekly_hours=int(data.get("weekly_hours", 5)),
        topics=topics,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The LangGraph node
#
# This is what LangGraph calls. It receives the full state dict and
# returns a partial update dict with only the keys it changed.
# ─────────────────────────────────────────────────────────────────────────────

def curriculum_planner_node(
    state: dict, config: RunnableConfig | None = None
) -> dict:
    """
    LangGraph node: Curriculum Planner

    Reads:
        state["goal"]: the user's learning goal

    Writes:
        state["roadmap"]  : populated StudyRoadmap on success
        state["messages"]: the LLM messages (prompt + response)
        state["error"]    : error message string on failure, None on success

    Flow:
        1. Extract goal from state
        2. Build LLM messages (system prompt + human message)
        3. Call Ollama
        4. Parse JSON response into StudyRoadmap
        5. Return partial state update
    """
    goal = state.get("goal", "").strip()

    if not goal:
        return {"error": "No learning goal provided. Set state['goal'] before running."}

    print(f"\n[Curriculum Planner] Building roadmap for: '{goal}'")

    messages = [
        HumanMessage(content=(
            f"Create a study roadmap for this learning goal: {goal}"
            f"{planner_research(goal)}"
        )),
    ]

    print(f"[Curriculum Planner] Calling {state.get('model_provider', 'ollama')}")
    try:
        callbacks = (config or {}).get("callbacks")
        invoke_config = {"callbacks": callbacks} if callbacks else None
        agent = build_planner_agent(
            state.get("model_provider", "ollama"),
            state.get("model_name", ""),
        )
        result = (
            agent.invoke({"messages": messages}, config=invoke_config)
            if invoke_config else agent.invoke({"messages": messages})
        )
        response = result["messages"][-1]
        print(f"[Curriculum Planner] LLM output:\n{response.content}")
    except Exception as e:
        print(f"[Curriculum Planner] LLM call failed: {e}")
        return {
            "error": f"LLM call failed: {e}",
            "messages": messages,
        }

    # response is an AIMessage object
    # response.content is the string the model returned
    try:
        roadmap = parse_roadmap_json(response.content)
    except ValueError as e:
        print(f"[Curriculum Planner] Parse error: {e}")
        # Return error state, the graph can route to an error handler
        return {
            "error": str(e),
            "messages": messages + [response],
        }

    # Success, log what was created
    print(f"[Curriculum Planner] Created roadmap: {len(roadmap.topics)} topics, "
          f"{roadmap.total_weeks} weeks")
    for i, topic in enumerate(roadmap.topics, 1):
        prereqs = f" (needs: {', '.join(topic.prerequisites)})" if topic.prerequisites else ""
        print(f"  {i}. {topic.title}{prereqs}, {topic.estimated_minutes} min")

    # Return partial state update.
    # Only include the keys we changed.
    # LangGraph merges this into the full state.
    return {
        "roadmap": roadmap,
        "messages": messages + [response],
        "error": None,
    }
