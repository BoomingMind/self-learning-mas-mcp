"""
main.py

Entry point for the Learning Accelerator.
Runs an interactive study session, with optional Langfuse observability.

Usage:
  python main.py "Learn Python closures from scratch"
  python main.py --resume <session-id>
"""

import sys
import uuid
from pathlib import Path

# Add src/ to Python path before any project imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from langgraph.types import Command

from graph.workflow import graph
from graph.state import initial_state, StudyRoadmap, QuizResult
from observability.langfuse_setup import get_langfuse_config, flush_langfuse


def print_session_summary(result: dict) -> None:
    """Print a summary of the completed session."""
    # After SqliteSaver round-trip, roadmap and quiz_results may be plain dicts.
    # Coerce them back to dataclasses before accessing attributes.
    raw_roadmap = result.get("roadmap")
    if raw_roadmap is None:
        return

    roadmap = (
        StudyRoadmap.from_dict(raw_roadmap)
        if isinstance(raw_roadmap, dict)
        else raw_roadmap
    )

    raw_results = result.get("quiz_results", [])
    quiz_results = [
        QuizResult.from_dict(r) if isinstance(r, dict) else r
        for r in raw_results
    ]

    if not quiz_results:
        return

    print(f"\n{'='*60}")
    print("Session Summary")
    print(f"{'='*60}")
    print(f"Goal: {roadmap.goal}")
    print(f"Topics covered: {len(quiz_results)}/{len(roadmap.topics)}")

    total_score = sum(r.score for r in quiz_results)
    avg = total_score / len(quiz_results)
    print(f"Average score: {avg:.0%}\n")

    for r in quiz_results:
        status = "✓" if r.score >= 0.5 else "✗"
        weak = f", review: {', '.join(r.weak_areas)}" if r.weak_areas else ""
        print(f"  {status} {r.topic}: {r.score:.0%}{weak}")

    all_weak = result.get("weak_areas", [])
    if all_weak:
        print(f"\nTopics to revisit: {', '.join(all_weak)}")

    print(f"{'='*60}\n")


def run_session(goal: str, session_id: str | None = None) -> None:
    """Run a complete interactive study session with Langfuse tracing."""
    is_resume = session_id is not None
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    # get_langfuse_config() builds the full run config:
    #   - thread_id for SQLite checkpointing
    #   - Langfuse callback handler (if LANGFUSE_PUBLIC_KEY is set)
    config = get_langfuse_config(session_id)

    print(f"\n{'='*60}")
    print("Learning Accelerator")
    print(f"Session ID: {session_id}")
    if is_resume:
        print("Resuming existing session...")
    else:
        print(f"Goal: {goal}")
    print(f"{'='*60}")

    # For a new session, provide initial state.
    # For a resume, pass None, LangGraph loads from checkpoint.
    state = None if is_resume else initial_state(goal, session_id)

    try:
        result = graph.invoke(state, config=config)
    except Exception as e:
        if is_resume:
            print(f"\n[ERROR] Could not resume session '{session_id}': {e}")
            print("If the session ID is wrong or the checkpoint database has been deleted, start a new session instead.")
            return
        raise

    # ── Handle human-in-the-loop interrupt ────────────────────────────
    # When the graph hits interrupt(), it pauses and returns with
    # "__interrupt__" in the result. We collect user input and resume.
    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value

        # After SqliteSaver round-trip, the roadmap in the payload may be a dict.
        raw_roadmap = interrupt_payload.get("roadmap")
        roadmap = (
            StudyRoadmap.from_dict(raw_roadmap)
            if isinstance(raw_roadmap, dict)
            else raw_roadmap
        )

        # Display the roadmap for approval
        if roadmap:
            print(f"\n{'='*60}")
            print("Proposed Study Plan")
            print(f"{'='*60}")
            print(f"Goal: {roadmap.goal}")
            print(f"Duration: {roadmap.total_weeks} weeks @ "
                  f"{roadmap.weekly_hours} hrs/week\n")
            for i, topic in enumerate(roadmap.topics, 1):
                prereqs = (f" (needs: {', '.join(topic.prerequisites)})"
                           if topic.prerequisites else "")
                print(f"  {i}. {topic.title} "
                      f"({topic.estimated_minutes} min){prereqs}")
                print(f"     {topic.description}")

        print(f"\n{interrupt_payload.get('prompt', 'Continue?')}")
        user_input = input("> ").strip()

        # Resume the graph with the user's decision
        result = graph.invoke(Command(resume=user_input), config=config)

    # ── Handle errors ─────────────────────────────────────────────────
    if result.get("error"):
        print(f"\n[ERROR] {result['error']}")
        return

    print_session_summary(result)

    # Flush Langfuse before exiting so all traces are sent
    flush_langfuse()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Learning Accelerator: a four-agent study system that plans a "
            "curriculum, explains topics from your notes, quizzes you, and "
            "adapts based on results. All inference runs locally via Ollama."
        ),
        epilog=(
            "Examples:\n"
            "  python main.py \"Learn Python closures from scratch\"\n"
            "  python main.py --resume a3f1b2c4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "goal", nargs="?",
        default="Learn Python closures and decorators from scratch",
        help="What you want to learn (default: a Python closures starter goal)",
    )
    parser.add_argument(
        "--resume", metavar="SESSION_ID",
        help="Resume an existing session by its 8-char ID",
    )
    args = parser.parse_args()

    if args.resume:
        run_session(goal="", session_id=args.resume)
    else:
        run_session(goal=args.goal)
