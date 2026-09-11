"""
tests/test_explainer.py

Unit tests for the Explainer agent.

The MCP server boundary is covered by test_mcp_stdio_client.py. These tests
cover the explainer's state selection without starting an LLM.

Run: python -m pytest tests/test_explainer.py -v
"""

from graph.state import StudyRoadmap, Topic, initial_state, get_current_topic
from agents.explainer import (
    add_learning_resources,
    classify_explainer_status,
    determine_explainer_status,
)
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
    def test_learning_resources_are_real_tavily_urls(self, monkeypatch):
        async def search(*args, **kwargs):
            return {
                "results": [
                    {"url": "https://docs.python.org/3/tutorial/"},
                    {"url": "not-a-url"},
                ]
            }

        monkeypatch.setattr("agents.explainer.call_tool", search)
        result = add_learning_resources("Explain recursion.", "Python recursion")
        assert "https://docs.python.org/3/tutorial/" in result
        assert "not-a-url" not in result
        assert "Further learning" in result

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

    def test_llm_status_classifier_uses_model_decision(self, monkeypatch):
        class Response:
            content = '{"status":"NEEDS_REMEDIATION","quiz_requested":false,"awaiting_approval":false}'

        class Model:
            def invoke(self, prompt):
                return Response()

        monkeypatch.setattr("agents.explainer.build_chat_model", lambda *args, **kwargs: Model())
        assert classify_explainer_status(
            "Closures",
            "I think the inner function copies the variable.",
            "Let us clarify how closures retain bindings.",
            "CONTINUE",
            False,
            1,
            "openrouter",
        ) == ("NEEDS_REMEDIATION", False, False)

    def test_explicit_quiz_request_remains_hard_safeguard(self, monkeypatch):
        class Response:
            content = '{"status":"CONTINUE","quiz_requested":false,"awaiting_approval":false}'

        class Model:
            def invoke(self, prompt):
                return Response()

        monkeypatch.setattr("agents.explainer.build_chat_model", lambda *args, **kwargs: Model())
        assert classify_explainer_status(
            "Closures", "Quiz me now", "Explanation", "CONTINUE", False, 1, "openrouter"
        ) == ("READY_FOR_QUIZ", True, False)

    def test_explainer_node_exposes_current_topic_response(self, monkeypatch):
        from langchain_core.messages import AIMessage, HumanMessage
        from agents import explainer

        topic = Topic("Recursion", "Functions calling themselves", 30)
        state = initial_state("learn recursion", "session-1")
        state["roadmap"] = StudyRoadmap("learn recursion", 1, [topic])
        state["messages"] = [
            HumanMessage(content="Create a study roadmap for this learning goal: learn recursion"),
            AIMessage(content='{"goal":"learn recursion","topics":[{"title":"Recursion"}]}'),
        ]
        state["learner_message"] = ""
        callback = object()

        async def run_agent(*args, **kwargs):
            assert args[3] == ""
            assert kwargs["callbacks"] == [callback]
            return [AIMessage(content="Roadmap text"), AIMessage(content="Recursion explanation")], AIMessage(
                content="Recursion explanation"
            )

        monkeypatch.setattr(explainer, "_run_explainer_agent", run_agent)
        monkeypatch.setattr(
            explainer,
            "classify_explainer_status",
            lambda *args, **kwargs: ("CONTINUE", False, False),
        )
        monkeypatch.setattr(
            explainer,
            "add_learning_resources",
            lambda text, topic_title: text + "\n### Further learning\n- https://example.com",
        )

        result = explainer.explainer_node(
            state, config={"callbacks": [callback]}
        )
        assert result["explanation"].startswith("Recursion explanation")

    def test_streamlit_explanation_filter_rejects_roadmap_json(self):
        import streamlit_app

        roadmap = '{"goal":"Learn Python","topics":[{"title":"Basics"}]}'
        assert streamlit_app._is_roadmap_response(roadmap)
        assert not streamlit_app._is_roadmap_response("Recursion is a function calling itself.")
