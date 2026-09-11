"""
tests/test_explainer.py

Unit tests for the Explainer agent.

The MCP server boundary is covered by test_mcp_stdio_client.py. These tests
cover the explainer's state selection without starting an LLM.

Run: python -m pytest tests/test_explainer.py -v
"""

from graph.state import StudyRoadmap, Topic, initial_state, get_current_topic
from agents.explainer import determine_explainer_status
from graph.workflow import route_after_explainer


class TestGetCurrentTopic:
    """Tests for the get_current_topic helper used by the Explainer."""

    def _make_state(self, n_topics=3, index=0):
        topics = [Topic(f"Topic {i}", f"Desc {i}", 30) for i in range(n_topics)]
        state = initial_state("test goal", "session-1")
        state["roadmap"] = StudyRoadmap("test goal", 1, topics)
        state["current_topic_index"] = index
        return state

    def test_returns_first_topic_at_index_0(self):
        state = self._make_state(index=0)
        topic = get_current_topic(state)
        assert topic is not None
        assert topic.title == "Topic 0"

    def test_returns_correct_topic_at_any_index(self):
        state = self._make_state(n_topics=5, index=3)
        topic = get_current_topic(state)
        assert topic.title == "Topic 3"

    def test_returns_none_when_index_past_end(self):
        state = self._make_state(n_topics=3, index=3)
        assert get_current_topic(state) is None

    def test_returns_none_without_roadmap(self):
        state = initial_state("test", "session-1")
        assert get_current_topic(state) is None


class TestInteractiveExplainerPolicy:
    def test_explicit_quiz_request_routes_to_quiz(self):
        status, requested, awaiting = determine_explainer_status("Quiz me now")
        assert (status, requested, awaiting) == ("READY_FOR_QUIZ", True, False)
        assert route_after_explainer({
            "explainer_status": status,
            "quiz_requested": requested,
        }) == "quiz_generator"

    def test_confusion_requests_remediation(self):
        status, requested, awaiting = determine_explainer_status(
            "I am confused about the example"
        )
        assert status == "NEEDS_REMEDIATION"
        assert requested is False
        assert awaiting is False

    def test_understanding_asks_for_approval(self):
        status, requested, awaiting = determine_explainer_status(
            "That makes sense", iterations=1
        )
        assert status == "AWAITING_QUIZ_APPROVAL"
        assert requested is False
        assert awaiting is True

    def test_declining_approval_stays_in_explainer(self):
        assert determine_explainer_status(
            "Explain more first", awaiting_approval=True
        ) == ("CONTINUE", False, False)
