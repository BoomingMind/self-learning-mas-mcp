"""
streamlit_app.py

Streamlit web interface for the Learning Accelerator.

Runs the same LangGraph graph as main.py, only the I/O mechanism
changes. Instead of terminal input/output, the app uses Streamlit
widgets and session state.

Run:
    streamlit run streamlit_app.py

Architecture:
    The app is a state machine with six screens:
    GOAL_INPUT → ROADMAP_APPROVAL → EXPLAINING → QUIZZING → COACHING → COMPLETE

    A separate graph instance (ui_graph) is compiled with
    interrupt_before=["quiz_generator"] so the graph pauses before the
    quiz step and returns control to Streamlit. The UI handles quiz I/O
    directly (calling generate_questions and grade_answer), then injects
    the QuizResult into the checkpoint via graph.update_state() and
    resumes execution from progress_coach onward.

    This means:
    - Zero changes to quiz_generator_node or run_quiz()
    - The terminal interface (main.py) is completely unaffected
    - The LangGraph graph code is identical, only I/O changes
"""

import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

import uuid
import streamlit as st
from langgraph.types import Command

from graph.workflow import (
    build_graph,
    delete_persisted_session,
    list_persisted_sessions,
)
from graph.state import initial_state, StudyRoadmap, QuizResult, QuizQuestion
from observability.langfuse_setup import get_langfuse_config, flush_langfuse
from agents.quiz_generator import generate_questions, grade_answer
from agents.explainer import _run_explainer_agent, add_learning_resources
from model_config import configured_model
from mcp_client import call_tool


# ── Build a UI-specific graph with interrupt_before=["quiz_generator"] ────────
# This stops the graph before quiz_generator runs so the UI can handle
# quiz I/O without calling input() which would block Streamlit.
ui_graph = build_graph(
    interrupt_before=["quiz_generator"],
    interrupt_after=["progress_coach"],
)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Learning Accelerator",
    page_icon="🎓",
    layout="centered",
)


# ── Session state initialisation ──────────────────────────────────────────────

SESSION_UI_KEYS = (
    "screen", "session_id", "graph_config", "roadmap",
    "current_topic_index", "quiz_questions", "current_question_idx",
    "graded_answers", "current_quiz_missing_concepts", "quiz_results",
    "weak_areas", "explanation", "topic_title", "topic_description",
    "coaching_message", "coaching_encouragement", "coaching_recommendation",
    "coaching_review_focus", "study_buddy_assistance", "coaching_topic_index",
    "coaching_topic", "error", "goal", "explainer_turns", "model_provider",
    "model_name",
)

def init_state():
    defaults = {
        "screen": "GOAL_INPUT",
        "session_id": None,
        "graph_config": None,
        "roadmap": None,
        "current_topic_index": 0,
        "quiz_questions": [],
        "current_question_idx": 0,
        "graded_answers": [],
        "current_quiz_missing_concepts": [],
        "quiz_results": [],
        "weak_areas": [],
        "explanation": "",
        "topic_title": "",
        "topic_description": "",
        "coaching_message": "",
        "coaching_encouragement": "",
        "coaching_recommendation": "",
        "coaching_review_focus": [],
        "study_buddy_assistance": "",
        "coaching_topic_index": 0,
        "coaching_topic": "",
        "error": None,
        "goal": "",
        "explainer_turns": [],
        "model_provider": "ollama",
        "model_name": "",
        "learning_sessions": {},
        "new_goal_mode": False,
        "postgres_sessions_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _copy_graph_config_without_callbacks(value: dict) -> dict:
    """Copy serializable graph settings while excluding runtime callbacks."""
    return {
        config_key: copy.deepcopy(config_value)
        for config_key, config_value in value.items()
        if config_key != "callbacks"
    }


def _save_active_session() -> None:
    """Persist the active UI state so another goal can be opened."""
    session_id = st.session_state.get("session_id")
    if not session_id:
        return
    record = st.session_state.learning_sessions.setdefault(session_id, {})
    record["goal"] = st.session_state.get("goal", "")
    record["screen"] = st.session_state.get("screen", "GOAL_INPUT")
    snapshot: dict[str, object] = {}
    for key in SESSION_UI_KEYS:
        value = st.session_state.get(key)
        if key == "graph_config" and isinstance(value, dict):
            # Langfuse callbacks own threads and locks and cannot be copied.
            # They are recreated when the session is restored.
            value = _copy_graph_config_without_callbacks(value)
        else:
            value = copy.deepcopy(value)
        snapshot[key] = value
    record["state"] = snapshot


def _persist_explainer_history() -> None:
    """Write the visible Explainer transcript into the current checkpoint."""
    config = st.session_state.get("graph_config")
    if not config or not st.session_state.get("session_id"):
        return
    ui_graph.update_state(
        config,
        {"explainer_turns": copy.deepcopy(st.session_state.explainer_turns)},
        as_node="explainer",
    )


def _append_explainer_turn(question: str, response: str) -> None:
    """Persist a new chat turn in both the UI session and its checkpoint."""
    turns = list(st.session_state.get("explainer_turns", []))
    turns.append({"question": question, "response": response})
    st.session_state.explainer_turns = turns
    _persist_explainer_history()
    _save_active_session()


def _start_topic_conversation(title: str, response: str) -> None:
    """Start a fresh transcript when the learner changes topics."""
    st.session_state.explainer_turns = [{
        "question": f"Explain {title}",
        "response": response,
    }]
    _persist_explainer_history()
    _save_active_session()


def _restore_explainer_history_from_checkpoint() -> None:
    """Hydrate the visible transcript from the active graph checkpoint."""
    config = st.session_state.get("graph_config")
    if not config or not st.session_state.get("session_id"):
        return
    snapshot = ui_graph.get_state(config)
    persisted = snapshot.values.get("explainer_turns", [])
    if not isinstance(persisted, list):
        return
    local = st.session_state.get("explainer_turns", [])
    if persisted and len(persisted) >= len(local) and persisted != local:
        st.session_state.explainer_turns = copy.deepcopy(persisted)


def _register_active_session() -> None:
    session_id = st.session_state.session_id
    st.session_state.learning_sessions.setdefault(session_id, {})
    _save_active_session()


def _restore_session(session_id: str) -> None:
    """Restore a previously opened goal and its current UI screen."""
    _save_active_session()
    saved = st.session_state.learning_sessions[session_id].get("state", {})
    for key in SESSION_UI_KEYS:
        if key in saved:
            if key == "graph_config" and isinstance(saved[key], dict):
                st.session_state[key] = get_langfuse_config(
                    session_id,
                    extra_config=_copy_graph_config_without_callbacks(saved[key]),
                )
            else:
                st.session_state[key] = copy.deepcopy(saved[key])
    if not st.session_state.topic_title:
        title, description = get_topic_info(
            {
                "roadmap": st.session_state.roadmap,
                "current_topic_index": st.session_state.current_topic_index,
            },
            st.session_state.current_topic_index,
        )
        st.session_state.topic_title = title
        st.session_state.topic_description = description
    st.session_state.new_goal_mode = False
    st.session_state.error = None


def _remove_session(session_id: str) -> None:
    """Remove a goal from the sidebar and delete its persisted graph state."""
    asyncio.run(call_tool(
        "memory",
        "memory_delete_session",
        {"session_id": session_id},
    ))
    delete_persisted_session(ui_graph.checkpointer, session_id)
    st.session_state.learning_sessions.pop(session_id, None)

    if st.session_state.get("session_id") == session_id:
        st.session_state.session_id = None
        st.session_state.graph_config = None
        st.session_state.new_goal_mode = True
        st.session_state.screen = "GOAL_INPUT"


def _screen_for_persisted_state(state: dict) -> str:
    """Choose the UI screen that can resume a persisted graph state."""
    roadmap = state.get("roadmap")
    if roadmap and not state.get("approved", False):
        return "ROADMAP_APPROVAL"
    if state.get("coaching_summary"):
        return "COACHING"
    topics = roadmap.get("topics", []) if isinstance(roadmap, dict) else []
    if topics and state.get("current_topic_index", 0) >= len(topics):
        return "COMPLETE"
    return "EXPLAINING" if roadmap else "GOAL_INPUT"


def _load_postgres_sessions() -> None:
    """Populate the sidebar from checkpoints created by earlier app runs."""
    if st.session_state.postgres_sessions_loaded:
        return
    persisted = list_persisted_sessions(ui_graph.checkpointer)
    for session_id, state in persisted.items():
        if session_id in st.session_state.learning_sessions:
            continue
        state["screen"] = _screen_for_persisted_state(state)
        state["session_id"] = session_id
        title, description = get_topic_info(
            state,
            state.get("current_topic_index", 0),
        )
        if not state.get("topic_title"):
            state["topic_title"] = title
        if not state.get("topic_description"):
            state["topic_description"] = description
        state["coaching_message"] = state.get("coaching_summary", "")
        state["graph_config"] = _copy_graph_config_without_callbacks(
            get_langfuse_config(session_id)
        )
        st.session_state.learning_sessions[session_id] = {
            "goal": state.get("goal", "Untitled goal"),
            "screen": state["screen"],
            "state": state,
        }
    st.session_state.postgres_sessions_loaded = True


def render_session_sidebar() -> None:
    """Render a ChatGPT-style learning-goal switcher."""
    _save_active_session()
    with st.sidebar:
        st.title("Learning goals")
        if st.button("＋ New goal", use_container_width=True, type="primary"):
            new_session()
            st.rerun()

        sessions = st.session_state.learning_sessions
        if not sessions:
            st.caption("Create a goal to start a learning session.")
            return

        session_ids = list(sessions)
        active_id = st.session_state.get("session_id")
        for sid in session_ids:
            label = (sessions[sid].get("goal") or "Untitled goal")[:48]
            selector_col, remove_col = st.columns([5, 1])
            with selector_col:
                if st.button(
                    label,
                    key=f"select-session-{sid}",
                    use_container_width=True,
                    type="primary" if sid == active_id else "secondary",
                ):
                    if sid != active_id and not st.session_state.get(
                        "new_goal_mode", False
                    ):
                        _restore_session(sid)
                        st.rerun()
            with remove_col:
                if st.button(
                    "×",
                    key=f"remove-session-{sid}",
                    help=f"Remove {label}",
                ):
                    _remove_session(sid)
                    st.rerun()

def go_to(screen: str):
    st.session_state.screen = screen


def get_roadmap() -> StudyRoadmap | None:
    r = st.session_state.roadmap
    if r is None:
        return None
    if isinstance(r, dict):
        return StudyRoadmap.from_dict(r)
    return r


def extract_explanation(messages: list) -> str:
    """Get a topic explanation, never the Curriculum Planner roadmap JSON."""
    from langchain_core.messages import AIMessage
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            content = message_text(msg.content).strip()
            if _is_roadmap_response(content):
                continue
            return content
    return ""


def _is_roadmap_response(content: str) -> bool:
    """Identify planner output so it cannot leak into the Explainer screen."""
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and isinstance(parsed.get("topics"), list)


def extract_coaching(messages: list) -> str:
    """Get the latest coaching message."""
    from langchain_core.messages import AIMessage
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return message_text(msg.content)
    return ""


def message_text(content: Any) -> str:
    """Normalize LangChain text or content-block responses for Streamlit."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def get_topic_info(result: dict, idx: int):
    """Return (title, description) for topic at idx from result or session state."""
    roadmap = result.get("roadmap") or st.session_state.roadmap
    rm = roadmap
    if isinstance(rm, dict):
        rm = StudyRoadmap.from_dict(rm)
    if rm and idx < len(rm.topics):
        topic = rm.topics[idx]
        title = topic.title if hasattr(topic, "title") else topic.get("title", "")
        desc = topic.description if hasattr(topic, "description") else topic.get("description", "")
        return title, desc
    return "", ""


def new_session():
    _save_active_session()
    st.session_state.new_goal_mode = True
    defaults = {
        "screen": "GOAL_INPUT",
        "session_id": None,
        "graph_config": None,
        "roadmap": None,
        "current_topic_index": 0,
        "quiz_questions": [],
        "current_question_idx": 0,
        "graded_answers": [],
        "current_quiz_missing_concepts": [],
        "quiz_results": [],
        "weak_areas": [],
        "explanation": "",
        "topic_title": "",
        "topic_description": "",
        "coaching_message": "",
        "coaching_encouragement": "",
        "coaching_recommendation": "",
        "coaching_review_focus": [],
        "study_buddy_assistance": "",
        "coaching_topic_index": 0,
        "coaching_topic": "",
        "error": None,
        "goal": "",
        "explainer_turns": [],
        "model_provider": "ollama",
        "model_name": "",
    }
    for key, value in defaults.items():
        st.session_state[key] = value


# ── Graph interaction ─────────────────────────────────────────────────────────

def start_session(goal: str, model_provider: str = "ollama", model_name: str = ""):
    """
    Start a new session. Runs: curriculum_planner → human_approval (interrupt).
    """
    session_id = str(uuid.uuid4())[:8]
    config = get_langfuse_config(session_id)
    st.session_state.session_id = session_id
    st.session_state.graph_config = config
    st.session_state.goal = goal
    st.session_state.model_provider = model_provider
    st.session_state.model_name = model_name
    st.session_state.new_goal_mode = False
    _register_active_session()

    state = initial_state(
        goal, session_id,
        model_provider=model_provider,
        model_name=model_name,
    )

    try:
        with st.spinner("Building your study roadmap..."):
            result = ui_graph.invoke(state, config=config)
    except Exception as exc:
        st.session_state.error = str(exc)
        return

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.session_state.roadmap = payload.get("roadmap")
        _save_active_session()
        go_to("ROADMAP_APPROVAL")
    elif result.get("error"):
        st.session_state.error = result["error"]
    else:
        st.session_state.error = "Unexpected: no interrupt after planner."


def approve_roadmap(approved: bool):
    """
    Resume after roadmap decision.

    If approved:
        Graph runs: human_approval_node → explainer_node
        Then pauses at interrupt_before=["quiz_generator"]
        We extract explanation and generate quiz questions.

    If rejected:
        Graph runs: human_approval_node → curriculum_planner → interrupt
        New roadmap is shown.
    """
    decision = "yes" if approved else "no"

    try:
        with st.spinner("Starting your study session..." if approved else "Generating a new plan..."):
            result = ui_graph.invoke(
                Command(resume=decision),
                config=st.session_state.graph_config,
            )
    except Exception as exc:
        st.session_state.error = str(exc)
        return

    if "__interrupt__" in result:
        # Roadmap rejected, new plan generated
        payload = result["__interrupt__"][0].value
        st.session_state.roadmap = payload.get("roadmap")
        go_to("ROADMAP_APPROVAL")
        _save_active_session()
        return

    # Graph paused before quiz_generator, explainer has finished
    # result contains messages with the explanation
    messages = result.get("messages", [])
    # Prefer the explicit Explainer output. The checkpoint also contains the
    # planner's roadmap message, which must never be shown as the explanation.
    candidate = result.get("explanation", "")
    explanation = (
        str(candidate).strip()
        if candidate and not _is_roadmap_response(str(candidate))
        else extract_explanation(messages)
    )
    st.session_state.explanation = explanation

    roadmap = result.get("roadmap") or st.session_state.roadmap
    st.session_state.roadmap = roadmap
    idx = result.get("current_topic_index", 0)
    st.session_state.current_topic_index = idx

    title, desc = get_topic_info(result, idx)
    st.session_state.topic_title = title
    st.session_state.topic_description = desc
    _start_topic_conversation(title, explanation)

    # Keep the learner in the Explainer; quiz generation is explicit.
    go_to("EXPLAINING")
    _save_active_session()


def advance_after_quiz(quiz_result: QuizResult):
    """
    After the UI-handled quiz is complete:
    1. Inject the QuizResult into the checkpoint as if quiz_generator ran.
    2. Resume graph through progress_coach and pause for visible feedback.
    3. The learner chooses review or continuation before the next Explainer runs.
    """
    config = st.session_state.graph_config
    existing = st.session_state.quiz_results
    all_weak = list(set(st.session_state.weak_areas + quiz_result.weak_areas))

    try:
        # Tell LangGraph that quiz_generator has already run with this result.
        # This sets the checkpoint state as if quiz_generator_node returned normally.
        ui_graph.update_state(
            config,
            {
                "quiz_results": existing + [quiz_result],
                "weak_areas": all_weak,
                "roadmap": st.session_state.roadmap,
                "current_topic_index": st.session_state.current_topic_index,
                "error": None,
            },
            as_node="quiz_generator",
        )

        # Resume, runs progress_coach, then either explainer (next topic)
        # or END if all topics are done.
        # Because interrupt_before=["quiz_generator"], if there is a next topic,
        # the graph will pause again before quiz_generator for that topic.
        with st.spinner("Getting coaching feedback..."):
            result = ui_graph.invoke(None, config=config)
    except Exception as exc:
        st.session_state.error = str(exc)
        return

    # Progress Coach is an explicit UI step. Do not run the next Explainer
    # until the learner has seen and acted on both recommendations.
    st.session_state.coaching_message = result.get(
        "coaching_summary", extract_coaching(result.get("messages", []))
    )
    st.session_state.coaching_encouragement = result.get("coaching_encouragement", "")
    st.session_state.coaching_recommendation = result.get("coaching_recommendation", "")
    st.session_state.coaching_review_focus = result.get("coaching_review_focus", [])
    st.session_state.study_buddy_assistance = result.get("study_buddy_assistance", "")
    st.session_state.coaching_topic_index = result.get(
        "coaching_topic_index", st.session_state.current_topic_index - 1
    )
    st.session_state.coaching_topic = result.get(
        "coaching_topic", st.session_state.topic_title
    )

    # Update accumulated state
    st.session_state.quiz_results = result.get("quiz_results", existing + [quiz_result])
    st.session_state.weak_areas = result.get("weak_areas", all_weak)
    new_idx = result.get("current_topic_index", st.session_state.current_topic_index + 1)
    st.session_state.current_topic_index = new_idx
    st.session_state.roadmap = result.get("roadmap", st.session_state.roadmap)

    rm = get_roadmap()

    go_to("COACHING")
    _save_active_session()


def continue_after_coaching(review: bool = False):
    """Resume the graph after the learner chooses a coaching recommendation."""
    config = st.session_state.graph_config
    target_index = (
        st.session_state.coaching_topic_index
        if review
        else st.session_state.current_topic_index
    )
    try:
        resume_update = {
            "current_topic_index": target_index,
            "coaching_summary": "",
            "coaching_encouragement": "",
            "coaching_recommendation": "",
            "coaching_review_focus": [],
            "study_buddy_assistance": "",
            "coaching_topic": "",
        }
        ui_graph.update_state(
            config,
            resume_update,
            as_node="progress_coach",
        )
        with st.spinner("Preparing the next explanation..."):
            result = ui_graph.invoke(None, config=config)
    except Exception as exc:
        st.session_state.error = str(exc)
        return

    messages = result.get("messages", [])
    idx = result.get("current_topic_index", target_index)
    st.session_state.current_topic_index = idx
    st.session_state.roadmap = result.get("roadmap", st.session_state.roadmap)
    candidate = result.get("explanation", "")
    st.session_state.explanation = (
        str(candidate).strip()
        if candidate and not _is_roadmap_response(str(candidate))
        else extract_explanation(messages)
    )
    title, desc = get_topic_info(result, idx)
    st.session_state.topic_title = title
    st.session_state.topic_description = desc
    _start_topic_conversation(title, st.session_state.explanation)
    st.session_state.coaching_message = ""
    st.session_state.study_buddy_assistance = ""
    go_to("EXPLAINING")
    _save_active_session()


def screen_coaching():
    st.title("💬 Progress Coach Feedback")
    st.markdown(f"### {st.session_state.coaching_topic}")
    st.success(st.session_state.coaching_message or "Keep practicing this topic.")
    if st.session_state.coaching_encouragement:
        st.info(st.session_state.coaching_encouragement)
    if st.session_state.coaching_recommendation:
        st.markdown("### Recommendation")
        st.markdown(st.session_state.coaching_recommendation)
    if st.session_state.coaching_review_focus:
        st.markdown("### Review focus")
        for focus in st.session_state.coaching_review_focus:
            st.markdown(f"- {focus}")
    if st.session_state.study_buddy_assistance:
        st.markdown("### 🤝 Study Buddy Recommendation")
        st.markdown(st.session_state.study_buddy_assistance)

    rm = get_roadmap()
    is_last = rm is None or st.session_state.current_topic_index >= len(rm.topics)
    st.markdown("---")
    st.markdown("What would you like to do next?")
    review_label = "🔁 Review this topic again"
    next_label = "Continue to next topic →" if not is_last else "Finish session →"
    col1, col2 = st.columns(2)
    with col1:
        if st.button(review_label, type="primary", use_container_width=True):
            continue_after_coaching(review=True)
            st.rerun()
    with col2:
        if st.button(next_label, use_container_width=True):
            if is_last:
                flush_langfuse()
                go_to("COMPLETE")
            else:
                continue_after_coaching(review=False)
            st.rerun()


# ── Screens ───────────────────────────────────────────────────────────────────

def screen_goal_input():
    st.title("🎓 Learning Accelerator")
    st.markdown(
        "Enter a learning goal and the system will build a personalised "
        "study plan, explain each topic adaptively, and quiz you "
        "as you go with the selected model and optional remote tools."
    )

    with st.form("goal_form"):
        goal = st.text_input(
            "What do you want to learn?",
            placeholder="e.g. Learn Python closures and decorators from scratch",
        )
        provider = st.selectbox(
            "Model provider",
            ["ollama", "openrouter", "groq"],
            key="selected_model_provider",
            format_func=lambda value: value.title(),
        )
        configured = configured_model(provider)
        options = [configured] if configured else ["(no model configured)"]
        model = st.selectbox(
            "Model",
            options,
            key=f"selected_model_{provider}",
            help="Configured from your .env file.",
        )
        submitted = st.form_submit_button("Build Study Plan →", type="primary")

    if submitted:
        if not goal.strip():
            st.error("Please enter a learning goal.")
        else:
            if not configured:
                st.error(f"No model is configured for {provider}.")
            else:
                start_session(goal.strip(), provider, model)
            st.rerun()

    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        if st.button("Try again"):
            st.session_state.error = None
            st.rerun()


def screen_roadmap_approval():
    st.title("📋 Your Study Plan")
    rm = get_roadmap()

    if rm is None:
        st.error("No roadmap found.")
        if st.button("Start over"):
            new_session()
            st.rerun()
        return

    st.markdown(f"**Goal:** {rm.goal}")
    st.markdown(f"**Duration:** {rm.total_weeks} weeks @ {rm.weekly_hours} hrs/week")
    st.markdown("---")

    for i, topic in enumerate(rm.topics, 1):
        title = topic.title if hasattr(topic, "title") else topic.get("title", "")
        desc = topic.description if hasattr(topic, "description") else topic.get("description", "")
        mins = topic.estimated_minutes if hasattr(topic, "estimated_minutes") else topic.get("estimated_minutes", "?")
        prereqs = topic.prerequisites if hasattr(topic, "prerequisites") else topic.get("prerequisites", [])
        prereq_text = f" *(needs: {', '.join(prereqs)})*" if prereqs else ""
        st.markdown(f"**{i}. {title}**, {mins} min{prereq_text}")
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{desc}")

    st.markdown("---")
    st.markdown("Does this study plan look good?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, start studying", type="primary", use_container_width=True):
            approve_roadmap(True)
            st.rerun()
    with col2:
        if st.button("🔄 No, generate a different plan", use_container_width=True):
            approve_roadmap(False)
            st.rerun()


def screen_explaining():
    _restore_explainer_history_from_checkpoint()
    rm = get_roadmap()
    total = len(rm.topics) if rm else 1
    idx = st.session_state.current_topic_index

    st.progress(idx / total, text=f"Topic {idx + 1} of {total}")
    st.header(f"📖 {st.session_state.topic_title}")
    st.caption(st.session_state.topic_description)

    if st.session_state.coaching_message:
        st.info(f"💬 **Coach:** {st.session_state.coaching_message}")

    if st.session_state.explainer_turns:
        for turn in st.session_state.explainer_turns:
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                st.markdown(turn["response"])
    elif st.session_state.explanation:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.explanation)
    else:
        st.warning("No explanation available, starting quiz with topic context.")

    follow_up = st.chat_input(
        "Ask for a simpler explanation, example, or more detail..."
    )
    if follow_up:
        try:
            with st.spinner("Adapting the explanation..."):
                _, response = asyncio.run(_run_explainer_agent(
                    st.session_state.topic_title,
                    st.session_state.topic_description,
                    st.session_state.session_id or "unknown",
                    follow_up.strip(),
                    st.session_state.model_provider,
                    st.session_state.model_name,
                    callbacks=(
                        st.session_state.graph_config or {}
                    ).get("callbacks"),
                ))
            st.session_state.explanation = add_learning_resources(
                message_text(response.content),
                st.session_state.topic_title,
            )
            _append_explainer_turn(
                follow_up.strip(),
                st.session_state.explanation,
            )
            st.rerun()
        except Exception as exc:
            st.session_state.error = str(exc)
            st.rerun()
    st.markdown(f"**Ready to test your knowledge of *{st.session_state.topic_title}*?**")

    if st.button("Start Quiz →", type="primary"):
        try:
            with st.spinner("Generating a short adaptive quiz..."):
                st.session_state.quiz_questions = generate_questions(
                    st.session_state.topic_title,
                    st.session_state.explanation,
                    n=3,
                    model_provider=st.session_state.model_provider,
                    model_name=st.session_state.model_name,
                    callbacks=(
                        st.session_state.graph_config or {}
                    ).get("callbacks"),
                )
        except Exception as exc:
            st.session_state.error = str(exc)
            st.rerun()
        st.session_state.current_question_idx = 0
        st.session_state.graded_answers = []
        st.session_state.current_quiz_missing_concepts = []
        st.session_state.coaching_message = ""
        go_to("QUIZZING")
        st.rerun()


def screen_quizzing():
    questions = st.session_state.quiz_questions
    q_idx = st.session_state.current_question_idx
    total_q = len(questions)
    rm = get_roadmap()
    total_topics = len(rm.topics) if rm else 1
    topic_idx = st.session_state.current_topic_index

    st.progress(topic_idx / total_topics, text=f"Topic {topic_idx + 1} of {total_topics}")
    if total_q > 0:
        st.progress(q_idx / total_q, text=f"Question {q_idx + 1} of {total_q}")

    st.title(f"🧠 Quiz: {st.session_state.topic_title}")
    st.markdown("---")

    # Show already-graded answers
    for i, graded in enumerate(st.session_state.graded_answers):
        status = "✅" if graded.correct else "❌"
        with st.expander(f"{status} Q{i+1}: {graded.question[:80]}...", expanded=False):
            st.markdown(f"**Your answer:** {graded.user_answer}")
            st.markdown(f"**Score:** {graded.score:.0%}")
            st.markdown(f"**Feedback:** {graded.feedback}")

    # Current question
    if q_idx < total_q:
        q = questions[q_idx]
        question_text = q.get("question", "")
        difficulty = q.get("difficulty", "medium")

        st.markdown(f"**Question {q_idx + 1} [{difficulty}]:**")
        st.markdown(question_text)

        with st.form(f"answer_form_{q_idx}"):
            answer = st.text_area(
                "Your answer:",
                placeholder="Type your answer here...",
                height=120,
                key=f"answer_input_{q_idx}",
            )
            submitted = st.form_submit_button("Submit Answer →", type="primary")

        if submitted:
            user_answer = answer.strip() or "(no answer provided)"
            expected = q.get("expected_answer", "")

            try:
                with st.spinner("Grading your answer..."):
                    grade = grade_answer(
                        question_text, expected, user_answer,
                        model_provider=st.session_state.model_provider,
                        model_name=st.session_state.model_name,
                        callbacks=(
                            st.session_state.graph_config or {}
                        ).get("callbacks"),
                    )
            except Exception as exc:
                st.session_state.error = str(exc)
                st.rerun()

            graded_q = QuizQuestion(
                question=question_text,
                expected_answer=expected,
                user_answer=user_answer,
                correct=bool(grade.get("correct", False)),
                feedback=grade.get("feedback", ""),
                score=float(grade.get("score", 0.0)),
            )
            st.session_state.graded_answers.append(graded_q)
            # Capture the LLM's identified missing concept (short topic-area phrase)
            # rather than the full feedback sentence, so the "Topics to Revisit"
            # list in screen_complete shows useful labels.
            missing = grade.get("missing_concept", "").strip()
            if missing:
                st.session_state.current_quiz_missing_concepts.append(missing)
            st.session_state.current_question_idx = q_idx + 1
            st.rerun()

    else:
        # All questions done
        st.markdown("---")
        graded = st.session_state.graded_answers
        avg_score = sum(q.score for q in graded) / len(graded) if graded else 0.0
        # Deduplicated list of identified weak areas across all questions in this quiz
        weak_areas = list(dict.fromkeys(
            st.session_state.current_quiz_missing_concepts
        ))

        st.success("✅ Quiz complete!")
        st.metric("Your score", f"{avg_score:.0%}")

        quiz_result = QuizResult(
            topic=st.session_state.topic_title,
            questions=graded,
            score=avg_score,
            weak_areas=weak_areas,
        )

        if st.button("Continue →", type="primary"):
            advance_after_quiz(quiz_result)
            st.rerun()


def screen_complete():
    st.title("🎉 Session Complete!")
    st.markdown("---")

    rm = get_roadmap()
    quiz_results = st.session_state.quiz_results

    if rm:
        st.markdown(f"**Goal:** {rm.goal}")

    if quiz_results:
        avg = sum(
            (r.score if hasattr(r, "score") else r.get("score", 0))
            for r in quiz_results
        ) / len(quiz_results)
        st.metric("Overall Average", f"{avg:.0%}")
        st.markdown("---")
        st.markdown("### Results by Topic")
        for r in quiz_results:
            if isinstance(r, dict):
                r = QuizResult.from_dict(r)
            status = "✅" if r.score >= 0.5 else "❌"
            weak = f", review: {', '.join(r.weak_areas[:2])}" if r.weak_areas else ""
            st.markdown(f"{status} **{r.topic}**: {r.score:.0%}{weak}")

    if st.session_state.weak_areas:
        st.markdown("---")
        st.markdown("### Topics to Revisit")
        for w in st.session_state.weak_areas[:5]:
            st.markdown(f"- {w}")

    st.markdown("---")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    st.caption("Resume via terminal: `python main.py --resume <session-id>`")

    if st.button("🔄 Start a New Session", type="primary"):
        new_session()
        st.rerun()


# ── Error banner ──────────────────────────────────────────────────────────────

def display_error():
    if st.session_state.error:
        st.error(f"Something went wrong: {st.session_state.error}")
        if st.button("← Start over"):
            new_session()
            st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────

_load_postgres_sessions()
render_session_sidebar()
screen = st.session_state.screen

if screen == "GOAL_INPUT":
    screen_goal_input()
elif screen == "ROADMAP_APPROVAL":
    display_error()
    screen_roadmap_approval()
elif screen == "EXPLAINING":
    display_error()
    screen_explaining()
elif screen == "QUIZZING":
    display_error()
    screen_quizzing()
elif screen == "COACHING":
    display_error()
    screen_coaching()
elif screen == "COMPLETE":
    screen_complete()
else:
    st.error(f"Unknown screen: {screen}")
    if st.button("Reset"):
        new_session()
        st.rerun()
