"""
tests/test_state.py

Validation tests for the shared state definition.

These tests are fast and deterministic, they do NOT call Ollama.
They verify that the data structures work correctly before any
agents are built on top of them.

Run: python -m pytest tests/test_state.py -v
"""

import pytest
from graph.state import (
    Topic,
    StudyRoadmap,
    QuizQuestion,
    QuizResult,
    AgentState,
    initial_state,
    get_current_topic,
    get_latest_quiz_result,
    session_is_complete,
)


class TestTopic:
    """Tests for the Topic dataclass."""

    def test_topic_defaults(self):
        """Topic should have sensible defaults for optional fields."""
        topic = Topic(
            title="Closures",
            description="Understanding Python closures",
            estimated_minutes=60,
        )
        assert topic.status == "pending"
        assert topic.prerequisites == []

    def test_topic_serialization_round_trip(self):
        """Topic should survive a to_dict/from_dict round trip."""
        original = Topic(
            title="Decorators",
            description="Python decorator pattern",
            estimated_minutes=45,
            prerequisites=["Closures"],
            status="completed",
        )
        restored = Topic.from_dict(original.to_dict())
        assert restored.title == original.title
        assert restored.prerequisites == original.prerequisites
        assert restored.status == original.status

    def test_topic_status_values(self):
        """Verify expected status values can be set."""
        topic = Topic("Test", "desc", 30)
        for status in ("pending", "in_progress", "completed", "needs_review"):
            topic.status = status
            assert topic.status == status


class TestStudyRoadmap:
    """Tests for the StudyRoadmap dataclass."""

    def _make_roadmap(self) -> StudyRoadmap:
        """Helper: create a test roadmap with 3 topics."""
        return StudyRoadmap(
            goal="Learn Python closures",
            total_weeks=2,
            topics=[
                Topic("Functions", "Review functions", 45),
                Topic("Closures", "Understand closures", 60,
                      prerequisites=["Functions"]),
                Topic("Decorators", "Build decorators", 75,
                      prerequisites=["Closures"]),
            ],
        )

    def test_completed_count_starts_at_zero(self):
        roadmap = self._make_roadmap()
        assert roadmap.completed_count() == 0

    def test_completed_count_increments(self):
        roadmap = self._make_roadmap()
        roadmap.topics[0].status = "completed"
        roadmap.topics[1].status = "completed"
        assert roadmap.completed_count() == 2

    def test_is_complete_false_when_pending_topics(self):
        roadmap = self._make_roadmap()
        assert roadmap.is_complete() is False

    def test_is_complete_true_when_all_done(self):
        roadmap = self._make_roadmap()
        for topic in roadmap.topics:
            topic.status = "completed"
        assert roadmap.is_complete() is True

    def test_is_complete_true_with_needs_review(self):
        """needs_review also counts as done, student moved on."""
        roadmap = self._make_roadmap()
        roadmap.topics[0].status = "completed"
        roadmap.topics[1].status = "needs_review"
        roadmap.topics[2].status = "completed"
        assert roadmap.is_complete() is True

    def test_serialization_round_trip(self):
        roadmap = self._make_roadmap()
        restored = StudyRoadmap.from_dict(roadmap.to_dict())
        assert restored.goal == roadmap.goal
        assert len(restored.topics) == len(roadmap.topics)
        assert restored.topics[1].prerequisites == ["Functions"]


class TestQuizResult:
    """Tests for QuizResult dataclass."""

    def _make_result(self, score: float) -> QuizResult:
        return QuizResult(
            topic="Closures",
            questions=[
                QuizQuestion(
                    question="What is a closure?",
                    expected_answer="A nested function that captures outer variables.",
                    user_answer="A function inside a function",
                    correct=score >= 0.5,
                    score=score,
                )
            ],
            score=score,
            weak_areas=[] if score >= 0.75 else ["late binding"],
        )

    def test_passed_threshold(self):
        assert self._make_result(0.5).passed() is True
        assert self._make_result(0.49).passed() is False

    def test_strong_pass_threshold(self):
        assert self._make_result(0.75).strong_pass() is True
        assert self._make_result(0.74).strong_pass() is False

    def test_serialization(self):
        result = self._make_result(0.8)
        d = result.to_dict()
        assert d["topic"] == "Closures"
        assert d["score"] == 0.8
        assert len(d["questions"]) == 1


class TestInitialState:
    """Tests for the initial_state factory function."""

    def test_initial_state_has_all_required_keys(self):
        """Every key in AgentState must be present in initial_state output."""
        state = initial_state("Learn Python", "session-001")

        required_keys = [
            "messages", "session_id", "goal", "roadmap",
            "approved", "current_topic_index", "quiz_results",
            "weak_areas", "study_materials_path", "error",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_initial_state_defaults(self):
        state = initial_state("Learn Python", "session-001")
        assert state["messages"] == []
        assert state["roadmap"] is None
        assert state["approved"] is False
        assert state["current_topic_index"] == 0
        assert state["quiz_results"] == []
        assert state["weak_areas"] == []
        assert state["error"] is None

    def test_initial_state_captures_goal_and_session(self):
        state = initial_state("Learn decorators", "abc-123")
        assert state["goal"] == "Learn decorators"
        assert state["session_id"] == "abc-123"

    def test_custom_study_materials_path(self):
        state = initial_state("Learn Python", "s1", "/custom/path/notes")
        assert state["study_materials_path"] == "/custom/path/notes"


class TestStateHelpers:
    """Tests for get_current_topic, get_latest_quiz_result, session_is_complete."""

    def _make_state_with_roadmap(self, n_topics=3, current_index=0):
        """Helper: build a state dict with a roadmap."""
        topics = [
            Topic(f"Topic {i}", f"Description {i}", 30)
            for i in range(n_topics)
        ]
        roadmap = StudyRoadmap("Test goal", 1, topics)
        state = initial_state("Test goal", "test-session")
        state["roadmap"] = roadmap
        state["current_topic_index"] = current_index
        return state

    def test_get_current_topic_returns_correct_topic(self):
        state = self._make_state_with_roadmap(n_topics=3, current_index=1)
        topic = get_current_topic(state)
        assert topic is not None
        assert topic.title == "Topic 1"

    def test_get_current_topic_returns_none_when_complete(self):
        state = self._make_state_with_roadmap(n_topics=3, current_index=3)
        assert get_current_topic(state) is None

    def test_get_current_topic_returns_none_without_roadmap(self):
        state = initial_state("Test", "session-1")
        assert get_current_topic(state) is None

    def test_get_latest_quiz_result_none_when_empty(self):
        state = initial_state("Test", "session-1")
        assert get_latest_quiz_result(state) is None

    def test_get_latest_quiz_result_returns_last(self):
        state = initial_state("Test", "session-1")
        r1 = QuizResult("Topic 0", [], 0.6, [])
        r2 = QuizResult("Topic 1", [], 0.9, [])
        state["quiz_results"] = [r1, r2]
        assert get_latest_quiz_result(state).topic == "Topic 1"

    def test_session_is_complete_false_with_pending_topics(self):
        state = self._make_state_with_roadmap(n_topics=3, current_index=0)
        assert session_is_complete(state) is False

    def test_session_is_complete_true_when_index_past_end(self):
        state = self._make_state_with_roadmap(n_topics=3, current_index=3)
        assert session_is_complete(state) is True

    def test_session_is_complete_true_without_roadmap(self):
        state = initial_state("Test", "session-1")
        assert session_is_complete(state) is True


def test_quiz_result_from_dict():
    """QuizResult.from_dict() reconstructs correctly from a plain dict."""
    data = {
        "topic": "Python Closures",
        "score": 0.75,
        "weak_areas": ["nonlocal keyword", "late binding"],
        "timestamp": "2026-04-19T10:00:00+00:00",
        "questions": [],
    }
    result = QuizResult.from_dict(data)
    assert result.topic == "Python Closures"
    assert result.score == 0.75
    assert result.weak_areas == ["nonlocal keyword", "late binding"]
    assert result.passed() is True
    assert result.strong_pass() is True


def test_get_latest_quiz_result_handles_dict():
    """get_latest_quiz_result() returns QuizResult even when state has raw dicts."""
    state = {
        "quiz_results": [
            {
                "topic": "Closures",
                "score": 0.6,
                "weak_areas": [],
                "timestamp": "",
                "questions": [],
            }
        ]
    }
    result = get_latest_quiz_result(state)
    assert result is not None
    assert isinstance(result, QuizResult)
    assert result.topic == "Closures"
    assert result.score == 0.6
