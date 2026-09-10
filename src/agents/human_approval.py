"""
src/agents/human_approval.py

Human-in-the-loop approval node.

This node sits between the Curriculum Planner and the Explainer.
It pauses the graph and waits for the user to approve (or reject)
the study roadmap before any studying begins.

How interrupt() works:
  1. LangGraph reaches this node
  2. interrupt() is called, graph execution PAUSES here
  3. LangGraph saves a full checkpoint to SQLite
  4. Control returns to the caller (main.py)
  5. main.py shows the roadmap and collects user input
  6. main.py calls graph.invoke(Command(resume=user_input), config)
  7. Execution resumes HERE with decision = user_input
  8. Node returns state update based on decision

Why this matters for production:
  In a web app, steps 4-6 would be:
    4. HTTP response returned to browser with the roadmap
    5. User submits a form
    6. New HTTP request resumes the graph
  The LangGraph code is identical in both terminal and web mode.
  Only the input collection mechanism changes.
"""

from langgraph.types import interrupt

from graph.state import StudyRoadmap


def human_approval_node(state: dict) -> dict:
    """
    LangGraph node: Human Approval

    Reads:
        state["roadmap"]  : the study plan to show the user

    Writes:
        state["approved"]: True if user approved, False if rejected

    When approved=False, the conditional edge routes back to the
    Curriculum Planner to generate a new roadmap.
    When approved=True, the graph continues to the Explainer.
    """
    roadmap: StudyRoadmap | None = state.get("roadmap")

    if roadmap is None:
        # No roadmap to approve, auto-approve and continue
        print("[Human Approval] No roadmap found, skipping approval")
        return {"approved": True}

    print("\n[Human Approval] Pausing for roadmap review...")

    # interrupt() pauses the graph here.
    # The dict passed to interrupt() is the "payload".
    # main.py reads this to know what to show the user.
    # Execution resumes when Command(resume=...) is called.
    decision = interrupt({
        "type": "roadmap_approval",
        "roadmap": roadmap,
        "prompt": (
            "Does this study plan look good?\n"
            "  Type 'yes' to start studying\n"
            "  Type 'no' to generate a different plan"
        ),
    })

    # decision is whatever the user typed (via Command(resume=...))
    approved = str(decision).lower().strip() in ("yes", "y", "ok", "approve")

    if approved:
        print("[Human Approval] Roadmap approved, starting study session")
    else:
        print("[Human Approval] Roadmap rejected, regenerating...")

    # LangGraph 1.1.0 does not fully carry pre-interrupt checkpoint state
    # into the next node after Command(resume=...). We return the full
    # state explicitly so downstream agents receive roadmap, session_id, etc.
    return {
        "approved": approved,
        "roadmap": roadmap,
        "goal": state.get("goal", ""),
        "session_id": state.get("session_id", ""),
        "current_topic_index": state.get("current_topic_index", 0),
        "quiz_results": state.get("quiz_results", []),
        "weak_areas": state.get("weak_areas", []),
        "study_materials_path": state.get("study_materials_path", "study_materials/sample_notes"),
        "error": None,
    }
