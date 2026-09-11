"""Unit tests for the generic A2A transport used by Study Buddy."""

import json
from unittest.mock import MagicMock, patch


class TestDiscoverAgent:
    @patch("a2a_services.a2a_client.httpx.get")
    def test_returns_card_on_success(self, mock_get):
        from a2a_services.a2a_client import discover_agent

        response = MagicMock()
        response.json.return_value = {"name": "Study Buddy"}
        response.raise_for_status = MagicMock()
        mock_get.return_value = response

        assert discover_agent("http://localhost:9002")["name"] == "Study Buddy"

    @patch("a2a_services.a2a_client.httpx.get")
    def test_returns_empty_dict_on_connection_error(self, mock_get):
        import httpx
        from a2a_services.a2a_client import discover_agent

        mock_get.side_effect = httpx.ConnectError("Connection refused")
        assert discover_agent("http://localhost:9002") == {}


class TestSendTask:
    @patch("a2a_services.a2a_client.httpx.post")
    def test_sends_current_message_envelope_and_parses_response(self, mock_post):
        from a2a_services.a2a_client import send_task

        response = MagicMock()
        response.json.return_value = {
            "result": {
                "kind": "message",
                "messageId": "response-1",
                "parts": [{
                    "kind": "text",
                    "text": json.dumps({"status": "complete"}),
                }],
            }
        }
        response.raise_for_status = MagicMock()
        mock_post.return_value = response

        result = send_task("http://localhost:9002", "{}")
        assert result["status"] == "complete"
        request = mock_post.call_args.kwargs["json"]
        assert request["method"] == "message/send"
        assert request["params"]["message"]["messageId"]

    @patch("a2a_services.a2a_client.httpx.post")
    def test_returns_error_on_connection_refused(self, mock_post):
        import httpx
        from a2a_services.a2a_client import send_task

        mock_post.side_effect = httpx.ConnectError("Connection refused")
        result = send_task("http://localhost:9002", "{}")
        assert "error" in result
        assert "connect" in result["error"].lower()

    @patch("a2a_services.a2a_client.httpx.post")
    def test_returns_error_on_timeout(self, mock_post):
        import httpx
        from a2a_services.a2a_client import send_task

        mock_post.side_effect = httpx.TimeoutException("Timed out")
        result = send_task("http://localhost:9002", "{}", timeout=1.0)
        assert "error" in result
        assert "timed out" in result["error"].lower()
