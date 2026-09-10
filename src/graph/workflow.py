import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agents.curriculum_planner import curriculum_planner_node
from agents.explainer import explainer_node
from agents.human_approval import human_approval_node
from agents.progress_coach import progress_coach_node
from agents.quiz_generator import quiz_generator_node
from graph.state import AgentState, session_is_complete


def route_after_approval(state: dict) -> str:
    if state.get("approved", False):
        return "explainer"
    return "curriculum_planner"


def route_after_coach(state: dict) -> str:
    if session_is_complete(state):
        return "end"
    return "explainer"


def build_graph(
    db_path: str = "data/checkpoints.db",
    interrupt_before: list | None = None,
):
    """
    Build and compile the Learning Accelerator graph.

    Args:
        db_path:          Path to the SQLite checkpoint database.
        interrupt_before: Optional list of node names to pause before.
                          Used by the Streamlit UI to intercept quiz_generator.
    """
    Path("data").mkdir(exist_ok=True)
    if db_path == "data/checkpoints.db":
        db_path = os.getenv("CHECKPOINT_DB", db_path)

    builder = StateGraph(AgentState)

    builder.add_node("curriculum_planner", curriculum_planner_node)
    builder.add_node("human_approval",     human_approval_node)
    builder.add_node("explainer",          explainer_node)
    builder.add_node("quiz_generator",     quiz_generator_node)
    builder.add_node("progress_coach",     progress_coach_node)

    builder.add_edge(START, "curriculum_planner")
    builder.add_edge("curriculum_planner", "human_approval")
    builder.add_edge("explainer",          "quiz_generator")
    builder.add_edge("quiz_generator",     "progress_coach")

    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"explainer": "explainer", "curriculum_planner": "curriculum_planner"},
    )
    builder.add_conditional_edges(
        "progress_coach",
        route_after_coach,
        {"explainer": "explainer", "end": END},
    )

    # CRITICAL: Create the connection directly. Do NOT use a context manager.
    # The connection must stay open for the process lifetime.
    # SqliteSaver requires check_same_thread=False because LangGraph runs
    # node functions and checkpoint writes on different threads.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )


graph = build_graph()