from langgraph.types import interrupt
from graph.state import StudyRoadmap


def human_approval_node(state: dict) -> dict:
    """
    LangGraph node: Human Approval

    Reads:  state["roadmap"]
    Writes: state["approved"]: True if approved, False if rejected.
            Also returns all other state keys explicitly (see note below).

    When approved=False, the conditional edge routes back to the
    Curriculum Planner to generate a new roadmap.
    When approved=True, the graph continues to the Explainer.
    """
    roadmap = state.get("roadmap")

    if roadmap is None:
        return {"approved": True}

    print(f"\n[Human Approval] Pausing for roadmap review...")

    # interrupt() pauses execution here.
    # The dict passed to interrupt() is the payload. The caller reads this
    # to know what to display to the user.
    # Execution resumes when Command(resume=value) is called by the caller.
    decision = interrupt({
        "type":   "roadmap_approval",
        "roadmap": roadmap,
        "prompt": (
            "Does this study plan look good?\n"
            "  Type 'yes' to start studying\n"
            "  Type 'no' to generate a different plan"
        ),
    })

    approved = str(decision).lower().strip() in ("yes", "y", "ok", "approve")

    if approved:
        print(f"[Human Approval] Roadmap approved. Starting study session.")
    else:
        print(f"[Human Approval] Roadmap rejected. Regenerating...")

    # LangGraph 1.1.0: after Command(resume=...), the next node receives only
    # the keys returned by this node. Not the full pre-interrupt checkpoint.
    # Returning the complete state explicitly ensures downstream agents
    # (explainer, quiz_generator, progress_coach) receive roadmap, session_id, etc.
    return {
        "approved":              approved,
        "roadmap":               roadmap,
        "goal":                  state.get("goal", ""),
        "session_id":            state.get("session_id", ""),
        "current_topic_index":   state.get("current_topic_index", 0),
        "quiz_results":          state.get("quiz_results", []),
        "weak_areas":            state.get("weak_areas", []),
        "study_materials_path":  state.get("study_materials_path",
                                           "study_materials/sample_notes"),
        "error":                 None,
    }