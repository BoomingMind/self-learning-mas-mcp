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

from graph.workflow import build_graph
from graph.state import initial_state, StudyRoadmap, QuizResult, QuizQuestion
from observability.langfuse_setup import get_langfuse_config, flush_langfuse
from agents.quiz_generator import generate_questions, grade_answer
from agents.explainer import _run_explainer_agent, add_learning_resources
from model_config import configured_model


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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    st.session_state.explainer_turns = [{
        "question": f"Explain {title}",
        "response": explanation,
    }]

    # Keep the learner in the Explainer; quiz generation is explicit.
    go_to("EXPLAINING")


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


def continue_after_coaching(review: bool = False):
    """Resume the graph after the learner chooses a coaching recommendation."""
    config = st.session_state.graph_config
    target_index = (
        st.session_state.coaching_topic_index
        if review
        else st.session_state.current_topic_index
    )
    try:
        if review:
            ui_graph.update_state(
                config,
                {"current_topic_index": target_index},
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
    st.session_state.explainer_turns = [{
        "question": f"Explain {title}",
        "response": st.session_state.explanation,
    }]
    st.session_state.coaching_message = ""
    st.session_state.study_buddy_assistance = ""
    go_to("EXPLAINING")


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
    rm = get_roadmap()
    total = len(rm.topics) if rm else 1
    idx = st.session_state.current_topic_index

    st.progress(idx / total, text=f"Topic {idx + 1} of {total}")
    st.title(f"📖 {st.session_state.topic_title}")
    st.caption(st.session_state.topic_description)
    st.markdown("---")

    if st.session_state.coaching_message:
        st.info(f"💬 **Coach:** {st.session_state.coaching_message}")
        st.markdown("---")

    if st.session_state.explainer_turns:
        st.markdown("### Conversation")
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

    st.markdown("---")
    with st.form("explainer_follow_up"):
        follow_up = st.text_area(
            "Ask a follow-up",
            placeholder="Ask for a simpler explanation, example, or more detail...",
        )
        ask = st.form_submit_button("Ask Explainer")
    if ask:
        if not follow_up.strip():
            st.warning("Please enter a follow-up question.")
        else:
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
                st.session_state.explainer_turns.append({
                    "question": follow_up.strip(),
                    "response": st.session_state.explanation,
                })
                st.rerun()
            except Exception as exc:
                st.session_state.error = str(exc)
                st.rerun()
    st.markdown("---")
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
