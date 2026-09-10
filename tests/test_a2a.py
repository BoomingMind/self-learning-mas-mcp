"""
tests/test_a2a.py

Unit tests for A2A service components.

Tests validate:
  - Agent Card structure is correct
  - A2A client handles service-unavailable gracefully
  - Request/response parsing logic
  - Progress Coach A2A delegation with fallback

No running A2A service required, all network calls are mocked.

Run: python -m pytest tests/test_a2a.py -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Agent Card tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentCard:
    """Verify the Quiz Agent Card is correctly structured."""

    def test_agent_card_has_required_fields(self):
        """Agent Card must have name, url, version, skills."""
        from a2a_services.quiz_service import QUIZ_AGENT_CARD

        assert QUIZ_AGENT_CARD.name
        assert QUIZ_AGENT_CARD.url
        assert QUIZ_AGENT_CARD.version
        assert QUIZ_AGENT_CARD.skills
        assert len(QUIZ_AGENT_CARD.skills) > 0

    def test_agent_card_url_is_localhost(self):
        """For local development, URL should point to localhost:9001."""
        from a2a_services.quiz_service import QUIZ_AGENT_CARD
        assert "9001" in QUIZ_AGENT_CARD.url

    def test_skill_has_required_fields(self):
        """Each skill must have id, name, description."""
        from a2a_services.quiz_service import QUIZ_AGENT_CARD
        skill = QUIZ_AGENT_CARD.skills[0]
        assert skill.id
        assert skill.name
        assert skill.description
        assert len(skill.description) > 20

    def test_skill_has_examples(self):
        """Skills should have examples to guide callers."""
        from a2a_services.quiz_service import QUIZ_AGENT_CARD
        skill = QUIZ_AGENT_CARD.skills[0]
        assert skill.examples
        assert len(skill.examples) >= 1

    def test_skill_id_is_correct(self):
        """Skill ID should match the documented value."""
        from a2a_services.quiz_service import QUIZ_AGENT_CARD
        assert QUIZ_AGENT_CARD.skills[0].id == "generate_and_grade_quiz"


# ─────────────────────────────────────────────────────────────────────────────
# A2A client tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscoverAgent:
    """Tests for the discover_agent function."""

    @patch("a2a_services.a2a_client.httpx.get")
    def test_returns_card_on_success(self, mock_get):
        """Successful fetch should return the parsed Agent Card dict."""
        from a2a_services.a2a_client import discover_agent

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "Test Agent",
            "url":  "http://localhost:9001/",
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = discover_agent("http://localhost:9001")
        assert result["name"] == "Test Agent"

    @patch("a2a_services.a2a_client.httpx.get")
    def test_returns_empty_dict_on_connection_error(self, mock_get):
        """Connection error should return {} not raise."""
        import httpx
        from a2a_services.a2a_client import discover_agent

        mock_get.side_effect = httpx.ConnectError("Connection refused")
        result = discover_agent("http://localhost:9001")
        assert result == {}

    @patch("a2a_services.a2a_client.httpx.get")
    def test_returns_empty_dict_on_timeout(self, mock_get):
        """Timeout should return {} not raise."""
        import httpx
        from a2a_services.a2a_client import discover_agent

        mock_get.side_effect = httpx.TimeoutException("Timed out")
        result = discover_agent("http://localhost:9001")
        assert result == {}


class TestSendTask:
    """Tests for the send_task function."""

    @patch("a2a_services.a2a_client.httpx.post")
    def test_returns_parsed_result_on_success(self, mock_post):
        """Successful task should return parsed result dict."""
        from a2a_services.a2a_client import send_task

        quiz_result = {
            "status":    "questions_ready",
            "topic":     "Test Topic",
            "questions": [],
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "artifacts": [{
                    "parts": [{"type": "text", "text": json.dumps(quiz_result)}]
                }]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = send_task("http://localhost:9001", json.dumps({"topic": "Test"}))
        assert result["status"] == "questions_ready"

    @patch("a2a_services.a2a_client.httpx.post")
    def test_returns_error_on_connection_refused(self, mock_post):
        """Connection refused should return error dict not raise."""
        import httpx
        from a2a_services.a2a_client import send_task

        mock_post.side_effect = httpx.ConnectError("Connection refused")
        result = send_task("http://localhost:9001", "{}")
        assert "error" in result
        assert "connect" in result["error"].lower()

    @patch("a2a_services.a2a_client.httpx.post")
    def test_returns_error_on_timeout(self, mock_post):
        """Timeout should return error dict not raise."""
        import httpx
        from a2a_services.a2a_client import send_task

        mock_post.side_effect = httpx.TimeoutException("Timed out")
        result = send_task("http://localhost:9001", "{}", timeout=1.0)
        assert "error" in result
        assert "timed out" in result["error"].lower()


class TestDelegateQuizTask:
    """Tests for the high-level delegate_quiz_task function."""

    @patch("a2a_services.a2a_client.send_task")
    def test_sends_correct_payload(self, mock_send):
        """delegate_quiz_task should send topic, explanation, answers."""
        from a2a_services.a2a_client import delegate_quiz_task

        mock_send.return_value = {"status": "graded", "score": 0.8}

        delegate_quiz_task(
            topic="Closures",
            explanation="A closure captures outer scope...",
            answers=["my answer"],
        )

        # Verify the payload contains the right keys
        call_args = mock_send.call_args
        payload = json.loads(call_args[0][1])   # second positional arg
        assert payload["topic"] == "Closures"
        assert "explanation" in payload
        assert "answers" in payload

    @patch("a2a_services.a2a_client.send_task")
    def test_empty_answers_sends_empty_list(self, mock_send):
        """None answers should be sent as empty list."""
        from a2a_services.a2a_client import delegate_quiz_task

        mock_send.return_value = {"status": "questions_ready"}
        delegate_quiz_task("Topic", "Explanation", answers=None)

        payload = json.loads(mock_send.call_args[0][1])
        assert payload["answers"] == []


class TestIsQuizServiceAvailable:
    """Tests for the service health check."""

    @patch("a2a_services.a2a_client.discover_agent")
    def test_returns_true_when_card_available(self, mock_discover):
        """Returns True when Agent Card is fetchable."""
        from a2a_services.a2a_client import is_quiz_service_available

        mock_discover.return_value = {"name": "Quiz Service"}
        assert is_quiz_service_available() is True

    @patch("a2a_services.a2a_client.discover_agent")
    def test_returns_false_when_service_down(self, mock_discover):
        """Returns False when Agent Card fetch fails."""
        from a2a_services.a2a_client import is_quiz_service_available

        mock_discover.return_value = {}
        assert is_quiz_service_available() is False


# ─────────────────────────────────────────────────────────────────────────────
# Progress Coach A2A integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProgressCoachA2ADelegation:
    """Tests for the A2A delegation logic in Progress Coach."""

    def test_a2a_disabled_returns_none(self):
        """When USE_A2A=false, delegation should return None immediately."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"USE_A2A_QUIZ": "false"}):
            from agents.progress_coach import try_a2a_quiz_delegation
            result = try_a2a_quiz_delegation("Topic", "Explanation", [])
            assert result is None

    @patch("a2a_services.a2a_client.is_quiz_service_available", return_value=False)
    def test_returns_none_when_service_unavailable(self, mock_available):
        """Returns None when quiz service is not reachable."""
        from agents.progress_coach import try_a2a_quiz_delegation

        result = try_a2a_quiz_delegation("Topic", "Explanation", [])
        assert result is None

    @patch("a2a_services.a2a_client.is_quiz_service_available", return_value=True)
    @patch("a2a_services.a2a_client.delegate_quiz_task")
    def test_returns_result_when_service_available(
        self, mock_delegate, mock_available
    ):
        """Returns A2A result dict when service is available."""
        from agents.progress_coach import try_a2a_quiz_delegation

        mock_delegate.return_value = {
            "status": "graded",
            "score":  0.85,
            "weak_areas": [],
        }

        result = try_a2a_quiz_delegation("Topic", "Explanation", ["answer1"])
        assert result is not None
        assert result["status"] == "graded"
        assert result["score"] == 0.85

    @patch("a2a_services.a2a_client.is_quiz_service_available", return_value=True)
    @patch("a2a_services.a2a_client.delegate_quiz_task")
    def test_returns_none_on_delegation_error(
        self, mock_delegate, mock_available
    ):
        """Returns None when delegation returns an error."""
        from agents.progress_coach import try_a2a_quiz_delegation

        mock_delegate.return_value = {"error": "Service crashed"}

        result = try_a2a_quiz_delegation("Topic", "Explanation", [])
        assert result is None
