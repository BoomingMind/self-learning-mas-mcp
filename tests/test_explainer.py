"""
tests/test_explainer.py

Unit tests for the Explainer agent.

The MCP server boundary is covered by test_mcp_stdio_client.py. These tests
cover the explainer's state selection without starting an LLM.

Run: python -m pytest tests/test_explainer.py -v
"""

from graph.state import StudyRoadmap, Topic, initial_state, get_current_topic


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
