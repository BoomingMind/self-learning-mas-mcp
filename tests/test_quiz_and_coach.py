"""
tests/test_quiz_and_coach.py

Unit tests for the Quiz Generator and Progress Coach agents.

Tests validate:
  - JSON parsing for question generation and grading
  - QuizResult construction
  - Progress Coach routing logic
  - State updates after coaching

No Ollama required, all LLM calls are tested via their
parsing functions, not the full inference loop.

Run: python -m pytest tests/test_quiz_and_coach.py -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

from graph.state import (
    QuizQuestion,
    QuizResult,
    StudyRoadmap,
    Topic,
    initial_state,
    get_latest_quiz_result,
    session_is_complete,
)
from agents.quiz_generator import generate_questions, grade_answer
from agents.progress_coach import progress_coach_node, PASS_THRESHOLD
from graph.workflow import route_after_coach


# ─────────────────────────────────────────────────────────────────────────────
# Quiz Generator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateQuestions:
    """Tests for question generation, validates fallback and parsing."""

    def _make_valid_json(self, n=3) -> str:
        questions = [
            {
                "question": f"Question {i}?",
                "expected_answer": f"Answer {i}",
                "difficulty": "medium",
            }
            for i in range(n)
        ]
        return json.dumps({"questions": questions})

    @patch("agents.quiz_generator.ChatOllama")
    def test_returns_questions_from_valid_json(self, mock_ollama):
        """Valid LLM JSON response should return a list of questions."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=self._make_valid_json(3))
        mock_ollama.return_value.bind_tools = MagicMock(return_value=mock_llm)
        mock_ollama.return_value = mock_llm

        result = generate_questions("Closures", "explanation text", n=3)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["question"] == "Question 0?"

    @patch("agents.quiz_generator.ChatOllama")
    def test_fallback_on_invalid_json(self, mock_ollama):
        """Invalid JSON should return one fallback question, not raise."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="this is not json {{{")
        mock_ollama.return_value = mock_llm

        result = generate_questions("Closures", "explanation", n=3)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "Closures" in result[0]["question"]

    @patch("agents.quiz_generator.ChatOllama")
    def test_fallback_on_missing_questions_key(self, mock_ollama):
        """JSON without 'questions' key should return fallback."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"something_else": []})
        )
        mock_ollama.return_value = mock_llm

        result = generate_questions("Topic", "explanation", n=2)
        assert len(result) == 1  # fallback


class TestGradeAnswer:
    """Tests for the grading function."""

    @patch("agents.quiz_generator.ChatOllama")
    def test_returns_grade_dict_from_valid_json(self, mock_ollama):
        """Valid grading JSON should return dict with expected keys."""
        grade_json = json.dumps({
            "correct": True,
            "score": 0.85,
            "feedback": "Good understanding shown.",
            "missing_concept": "",
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=grade_json)
        mock_ollama.return_value = mock_llm

        result = grade_answer("What is a closure?", "A function capturing...", "My answer")
        assert result["correct"] is True
        assert result["score"] == 0.85
        assert "feedback" in result

    @patch("agents.quiz_generator.ChatOllama")
    def test_safe_default_on_invalid_json(self, mock_ollama):
        """Invalid grading JSON should return safe default, not raise."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="not json")
        mock_ollama.return_value = mock_llm

        result = grade_answer("Q?", "Expected", "Student answer")
        assert "correct" in result
        assert "score" in result
        assert result["score"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Progress Coach tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterCoach:
    """Tests for the conditional routing function."""

    def _make_state(self, n_topics=3, current_index=0):
        topics = [Topic(f"Topic {i}", f"Desc {i}", 30) for i in range(n_topics)]
        state = initial_state("test", "s1")
        state["roadmap"] = StudyRoadmap("test", 1, topics)
        state["current_topic_index"] = current_index
        return state

    def test_routes_to_explainer_when_topics_remain(self):
        state = self._make_state(n_topics=3, current_index=0)
        assert route_after_coach(state) == "explainer"

    def test_routes_to_end_when_all_topics_done(self):
        state = self._make_state(n_topics=3, current_index=3)
        assert route_after_coach(state) == "end"

    def test_routes_to_end_at_exact_boundary(self):
        """Index == len(topics) should route to end."""
        state = self._make_state(n_topics=2, current_index=2)
        assert route_after_coach(state) == "end"

    def test_routes_to_end_without_roadmap(self):
        state = initial_state("test", "s1")
        assert route_after_coach(state) == "end"


class TestProgressCoachNode:
    """Tests for the progress_coach_node function."""

    def _make_state_with_quiz_result(self, score: float, n_topics=2, current_index=0):
        topics = [Topic(f"Topic {i}", f"Desc {i}", 30) for i in range(n_topics)]
        roadmap = StudyRoadmap("test goal", 1, topics)
        state = initial_state("test goal", "session-test")
        state["roadmap"] = roadmap
        state["current_topic_index"] = current_index
        state["quiz_results"] = [
            QuizResult(
                topic=topics[current_index].title,
                questions=[],
                score=score,
                weak_areas=["late binding"] if score < 0.5 else [],
            )
        ]
        return state

    @patch("agents.progress_coach.get_coaching_message")
    def test_increments_topic_index_on_pass(self, mock_coaching):
        """A passing score should increment current_topic_index."""
        mock_coaching.return_value = {
            "summary": "Great job!",
            "encouragement": "Keep it up!",
        }
        state = self._make_state_with_quiz_result(score=0.8, n_topics=3, current_index=0)
        result = progress_coach_node(state)
        assert result["current_topic_index"] == 1

    @patch("agents.progress_coach.get_coaching_message")
    def test_increments_index_on_fail_too(self, mock_coaching):
        """Even a failing score increments the index (move on policy)."""
        mock_coaching.return_value = {
            "summary": "Keep practicing!",
            "encouragement": "You'll get it!",
        }
        state = self._make_state_with_quiz_result(score=0.3, n_topics=3, current_index=0)
        result = progress_coach_node(state)
        assert result["current_topic_index"] == 1

    @patch("agents.progress_coach.get_coaching_message")
    def test_marks_topic_completed_on_pass(self, mock_coaching):
        """A passing score should mark the topic as 'completed'."""
        mock_coaching.return_value = {"summary": ".", "encouragement": "."}
        state = self._make_state_with_quiz_result(score=0.75, n_topics=2, current_index=0)
        result = progress_coach_node(state)
        assert result["roadmap"].topics[0].status == "completed"

    @patch("agents.progress_coach.get_coaching_message")
    def test_marks_topic_needs_review_on_fail(self, mock_coaching):
        """A failing score should mark the topic as 'needs_review'."""
        mock_coaching.return_value = {"summary": ".", "encouragement": "."}
        state = self._make_state_with_quiz_result(score=0.3, n_topics=2, current_index=0)
        result = progress_coach_node(state)
        assert result["roadmap"].topics[0].status == "needs_review"

    @patch("agents.progress_coach.get_coaching_message")
    def test_returns_coaching_message(self, mock_coaching):
        """Node should return a message with the coaching summary."""
        mock_coaching.return_value = {
            "summary": "You scored 80%!",
            "encouragement": "Keep going!",
        }
        state = self._make_state_with_quiz_result(score=0.8, n_topics=2)
        result = progress_coach_node(state)
        messages = result.get("messages", [])
        assert len(messages) > 0
        assert isinstance(messages[-1], AIMessage)

    def test_returns_error_without_quiz_results(self):
        """Node should return error state if no quiz results exist."""
        state = initial_state("test", "session-1")
        result = progress_coach_node(state)
        assert result.get("error") is not None


class TestPassThreshold:
    """Validate the PASS_THRESHOLD constant is correctly defined."""

    def test_threshold_is_between_0_and_1(self):
        assert 0.0 < PASS_THRESHOLD < 1.0

    def test_threshold_is_05(self):
        """Default threshold should be 0.5."""
        assert PASS_THRESHOLD == 0.5
