"""
tests/test_crewai_interop.py

Unit tests for the CrewAI Study Buddy and cross-framework A2A interop.

Tests validate:
  - Study Buddy Agent Card structure
  - CrewAI crew builder creates valid crew
  - A2A client study buddy helpers
  - Progress Coach study buddy integration with fallback

No running services required, all network calls are mocked.
CrewAI crew builder is tested structurally (not executed).

Run: python -m pytest tests/test_crewai_interop.py -v
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Study Buddy Agent Card tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStudyBuddyAgentCard:
    """Verify the Study Buddy Agent Card is correctly structured."""

    def test_agent_card_has_required_fields(self):
        from crewai_agent.study_buddy import STUDY_BUDDY_CARD
        assert STUDY_BUDDY_CARD.name
        assert STUDY_BUDDY_CARD.url
        assert STUDY_BUDDY_CARD.version
        assert STUDY_BUDDY_CARD.skills
        assert len(STUDY_BUDDY_CARD.skills) > 0

    def test_agent_card_url_is_port_9002(self):
        """Study Buddy uses port 9002, Quiz Service uses 9001."""
        from crewai_agent.study_buddy import STUDY_BUDDY_CARD
        assert "9002" in STUDY_BUDDY_CARD.url

    def test_skill_id_is_correct(self):
        from crewai_agent.study_buddy import STUDY_BUDDY_CARD
        assert STUDY_BUDDY_CARD.skills[0].id == "supplementary_study_assistance"

    def test_skill_mentions_crewai(self):
        """Agent Card should identify itself as a CrewAI agent."""
        from crewai_agent.study_buddy import STUDY_BUDDY_CARD
        card_text = (
            STUDY_BUDDY_CARD.description +
            STUDY_BUDDY_CARD.skills[0].description
        ).lower()
        assert "crewai" in card_text

    def test_different_port_from_quiz_service(self):
        """Study Buddy and Quiz Service must be on different ports."""
        from crewai_agent.study_buddy import STUDY_BUDDY_CARD
        from a2a_services.quiz_service import QUIZ_AGENT_CARD
        assert STUDY_BUDDY_CARD.url != QUIZ_AGENT_CARD.url


# ─────────────────────────────────────────────────────────────────────────────
# CrewAI crew builder tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildStudyBuddyCrew:
    """Tests for the CrewAI crew factory function."""

    def test_returns_crew_object(self):
        """build_study_buddy_crew should return a CrewAI Crew."""
        from crewai import Crew
        from crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew(
            topic="Python Closures",
            explanation="A closure captures variables...",
            weak_areas=["late binding"],
        )
        assert isinstance(crew, Crew)

    def test_crew_has_one_agent(self):
        """The study buddy crew should have exactly one agent."""
        from crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew("Topic", "Explanation", [])
        assert len(crew.agents) == 1

    def test_crew_has_one_task(self):
        """The study buddy crew should have exactly one task."""
        from crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew("Topic", "Explanation", [])
        assert len(crew.tasks) == 1

    def test_agent_has_study_buddy_role(self):
        """The agent's role should identify it as a Study Buddy."""
        from crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew("Topic", "Explanation", [])
        agent = crew.agents[0]
        assert "study" in agent.role.lower() or "buddy" in agent.role.lower()

    def test_task_description_contains_topic(self):
        """Task description should reference the topic."""
        from crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew(
            topic="Python Decorators",
            explanation="Decorators wrap functions...",
            weak_areas=[],
        )
        task_desc = crew.tasks[0].description
        assert "Python Decorators" in task_desc

    def test_task_description_contains_weak_areas(self):
        """Task description should reference specific weak areas."""
        from crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew(
            topic="Closures",
            explanation="A closure is...",
            weak_areas=["nonlocal keyword", "late binding"],
        )
        task_desc = crew.tasks[0].description
        assert "nonlocal keyword" in task_desc or "late binding" in task_desc

    def test_agent_has_topic_analyser_tool(self):
        """The study buddy agent should have the topic analyser tool."""
        from crewai_agent.study_buddy import build_study_buddy_crew, TopicAnalyserTool

        crew = build_study_buddy_crew("Topic", "Explanation", [])
        agent = crew.agents[0]
        tool_names = [type(t).__name__ for t in agent.tools]
        assert "TopicAnalyserTool" in tool_names

    def test_different_topics_create_different_tasks(self):
        """Two calls with different topics should produce different task descriptions."""
        from crewai_agent.study_buddy import build_study_buddy_crew

        crew1 = build_study_buddy_crew("Closures", "Explanation 1", [])
        crew2 = build_study_buddy_crew("Decorators", "Explanation 2", [])
        assert crew1.tasks[0].description != crew2.tasks[0].description


# ─────────────────────────────────────────────────────────────────────────────
# TopicAnalyserTool tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTopicAnalyserTool:
    """Tests for the CrewAI tool used by the Study Buddy."""

    def test_returns_json_string(self):
        """Tool should return a JSON-parseable string."""
        from crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = tool._run(topic="Python Closures", weak_areas=["late binding"])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_result_contains_topic(self):
        """Result should reference the input topic."""
        from crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = json.loads(tool._run(topic="Decorators", weak_areas=[]))
        assert result["topic"] == "Decorators"

    def test_result_has_required_keys(self):
        """Result must have topic, focus_areas, suggested_approach, study_tip."""
        from crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = json.loads(tool._run(topic="Closures", weak_areas=["late binding"]))
        for key in ["topic", "focus_areas", "suggested_approach", "study_tip"]:
            assert key in result, f"Missing key: {key}"

    def test_weak_areas_appear_in_focus_areas(self):
        """Provided weak areas should appear in focus_areas."""
        from crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = json.loads(
            tool._run(topic="Closures", weak_areas=["late binding", "nonlocal"])
        )
        assert "late binding" in result["focus_areas"]
        assert "nonlocal" in result["focus_areas"]

    def test_empty_weak_areas_uses_fallback(self):
        """Empty weak areas should produce a sensible fallback focus."""
        from crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = json.loads(tool._run(topic="Python Basics", weak_areas=[]))
        assert len(result["focus_areas"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# A2A client study buddy helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestStudyBuddyClient:
    """Tests for the study buddy A2A client helpers."""

    @patch("a2a_services.a2a_client.send_task")
    def test_request_study_assistance_sends_correct_payload(self, mock_send):
        """Client should send topic, explanation, and weak_areas."""
        from a2a_services.a2a_client import request_study_assistance

        mock_send.return_value = {
            "source": "crewai_study_buddy",
            "assistance": "Here is a fresh analogy...",
            "status": "complete",
        }

        request_study_assistance(
            topic="Closures",
            explanation="A closure is...",
            weak_areas=["late binding"],
        )

        payload = json.loads(mock_send.call_args[0][1])
        assert payload["topic"] == "Closures"
        assert payload["weak_areas"] == ["late binding"]
        assert "explanation" in payload

    @patch("a2a_services.a2a_client.discover_agent")
    def test_is_study_buddy_available_true(self, mock_discover):
        from a2a_services.a2a_client import is_study_buddy_available

        mock_discover.return_value = {"name": "CrewAI Study Buddy"}
        assert is_study_buddy_available() is True

    @patch("a2a_services.a2a_client.discover_agent")
    def test_is_study_buddy_available_false(self, mock_discover):
        from a2a_services.a2a_client import is_study_buddy_available

        mock_discover.return_value = {}
        assert is_study_buddy_available() is False


# ─────────────────────────────────────────────────────────────────────────────
# Progress Coach study buddy integration
# ─────────────────────────────────────────────────────────────────────────────

class TestProgressCoachStudyBuddy:
    """Tests for the Progress Coach → CrewAI Study Buddy integration."""

    def test_disabled_by_env_var_returns_none(self):
        """USE_STUDY_BUDDY=false should return None immediately."""
        with patch.dict(os.environ, {"USE_STUDY_BUDDY": "false"}):
            from agents.progress_coach import try_study_buddy_assistance
            result = try_study_buddy_assistance("Topic", "Explanation", [])
            assert result is None

    @patch("a2a_services.a2a_client.is_study_buddy_available", return_value=False)
    def test_returns_none_when_service_down(self, mock_avail):
        """Returns None when Study Buddy service is unreachable."""
        from agents.progress_coach import try_study_buddy_assistance
        result = try_study_buddy_assistance("Topic", "Explanation", [])
        assert result is None

    @patch("a2a_services.a2a_client.is_study_buddy_available", return_value=True)
    @patch("a2a_services.a2a_client.request_study_assistance")
    def test_returns_assistance_text_when_available(
        self, mock_assist, mock_avail
    ):
        """Returns assistance text when service responds successfully."""
        from agents.progress_coach import try_study_buddy_assistance

        mock_assist.return_value = {
            "source":     "crewai_study_buddy",
            "assistance": "Think of a closure like a backpack...",
            "status":     "complete",
        }

        result = try_study_buddy_assistance(
            topic="Closures",
            explanation="A closure is...",
            weak_areas=["late binding"],
        )
        assert result is not None
        assert "backpack" in result

    @patch("a2a_services.a2a_client.is_study_buddy_available", return_value=True)
    @patch("a2a_services.a2a_client.request_study_assistance")
    def test_returns_none_on_error_response(self, mock_assist, mock_avail):
        """Returns None when service returns an error."""
        from agents.progress_coach import try_study_buddy_assistance

        mock_assist.return_value = {
            "status": "error",
            "error": "CrewAI crashed",
        }
        result = try_study_buddy_assistance("Topic", "Explanation", [])
        assert result is None
