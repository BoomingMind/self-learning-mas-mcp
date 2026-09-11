"""
src/graph/workflow.py

LangGraph workflow definition for the Learning Accelerator.

Full graph:
  START → curriculum_planner → human_approval
        → (approved)  → explainer → quiz_generator → progress_coach
        → (more topics) → explainer
        → (all done)    → END
        → (rejected)    → curriculum_planner

Key design decisions:
  - PostgresSaver: checkpoints in the PostgreSQL container
  - interrupt() in human_approval_node: pauses to collect user approval
  - Routing functions are pure Python (no LLM calls in control flow)
  - All business logic lives in agents/, this file is wiring only
"""

import os
from typing import Any

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from agents.curriculum_planner import curriculum_planner_node
from agents.explainer import explainer_node
from agents.human_approval import human_approval_node
from agents.progress_coach import progress_coach_node
from agents.quiz_generator import quiz_generator_node
from graph.state import AgentState, session_is_complete

# Note: LangGraph deserializes dataclasses from PostgreSQL checkpoints as plain
# dicts. All state accessor functions in graph/state.py handle both dict and
# dataclass forms via isinstance checks and from_dict() classmethods.


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────────────────────────────────────

def route_after_approval(state: dict) -> str:
    """
    After the human approval node, decide what happens next.

    Returns:
        "explainer"         : user approved the plan, start studying
        "curriculum_planner", user rejected, generate a new plan
    """
    if state.get("approved", False):
        return "explainer"
    return "curriculum_planner"


def route_after_coach(state: dict) -> str:
    """
    After the progress coach, decide whether to continue or finish.

    Returns:
        "explainer", more topics remain
        "end"      : all topics have been covered
    """
    if session_is_complete(state):
        return "end"
    return "explainer"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_POSTGRES_URL = (
    "postgresql://postgres:postgres@127.0.0.1:5433/checkpoints"
)


def build_graph(
    postgres_url: str | None = None,
    interrupt_before: list | None = None,
):
    """
    Build and compile the Learning Accelerator graph.

    Args:
        postgres_url: PostgreSQL connection string for checkpoint storage.
        interrupt_before: List of node names to pause before (for UI integration).

    PostgresSaver persists checkpoints in PostgreSQL. The checkpoint tables
    are created automatically on startup.

    The human_approval_node uses interrupt() to pause execution
    and wait for user input before proceeding.
    """
    builder = StateGraph(AgentState)

    # ── Register all nodes ────────────────────────────────────────────
    builder.add_node("curriculum_planner", curriculum_planner_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("explainer", explainer_node)
    builder.add_node("quiz_generator", quiz_generator_node)
    builder.add_node("progress_coach", progress_coach_node)

    # ── Static edges ──────────────────────────────────────────────────
    builder.add_edge(START, "curriculum_planner")
    builder.add_edge("curriculum_planner", "human_approval")
    builder.add_edge("explainer", "quiz_generator")
    builder.add_edge("quiz_generator", "progress_coach")

    # ── Conditional edges ─────────────────────────────────────────────
    # After approval: start studying or regenerate plan
    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "explainer": "explainer",
            "curriculum_planner": "curriculum_planner",
        },
    )

    # After coaching: next topic or done
    builder.add_conditional_edges(
        "progress_coach",
        route_after_coach,
        {
            "explainer": "explainer",
            "end": END,
        },
    )

    # ── Compile with PostgreSQL checkpointer ──────────────────────────
    # Keep the connection open for the lifetime of the compiled graph.
    connection: Any = psycopg.connect(
        postgres_url
        or os.getenv("CHECKPOINT_POSTGRES_URL", DEFAULT_POSTGRES_URL),
        autocommit=True,
    )
    checkpointer = PostgresSaver(connection)
    checkpointer.setup()

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )
