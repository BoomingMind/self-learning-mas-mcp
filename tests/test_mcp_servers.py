"""
tests/test_mcp_servers.py

Tests for MCP server tools and resources.

These tests call the tool functions DIRECTLY as Python functions.
not through the MCP protocol. This is intentional:

1. Speed, no subprocess startup, no stdio piping overhead
2. Isolation, tests the logic, not the transport
3. Debuggability, stack traces point directly to the function

In production, the MCP transport (stdio or HTTP) is tested
separately via integration tests. For unit testing, calling
the functions directly is the right pattern.

Run: python -m pytest tests/test_mcp_servers.py -v
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_servers.memory_server import (
    memory_set,
    memory_get,
    memory_list_keys,
    memory_delete,
    memory_delete_session,
    get_session_summary,
)
import mcp_servers.memory_server as memory_server
from mcp_servers.onecompiler_server import execute_code
from mcp_servers.tavily_server import search_web, extract_web_page


# ─────────────────────────────────────────────────────────────────────────────
# Tavily server tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTavilyServer:
    """Verify the web-search MCP tool contract without external network calls."""

    @patch("mcp_servers.tavily_server._client")
    def test_search_web_uses_tavily_query_and_limits(self, mock_client):
        mock_client.return_value.search.return_value = {"results": [{"url": "https://example.com"}]}

        result = search_web(
            query="Python closures explained",
            max_results=3,
            search_depth="advanced",
            include_domains=["docs.python.org"],
            exclude_domains=["example.com"],
        )

        assert result["results"][0]["url"] == "https://example.com"
        mock_client.return_value.search.assert_called_once_with(
            query="Python closures explained",
            max_results=3,
            search_depth="advanced",
            include_domains=["docs.python.org"],
            exclude_domains=["example.com"],
        )

    @patch("mcp_servers.tavily_server._client")
    def test_extract_web_page_calls_tavily_extract_for_selected_url(self, mock_client):
        mock_client.return_value.extract.return_value = {"content": "excerpt"}

        result = extract_web_page("https://docs.python.org/3/library/collections.html")

        assert result["content"] == "excerpt"
        mock_client.return_value.extract.assert_called_once_with(urls=["https://docs.python.org/3/library/collections.html"])

    def test_search_web_requires_query(self):
        with pytest.raises(ValueError, match="query must not be empty"):
            search_web("   ")

    def test_extract_web_page_requires_url(self):
        with pytest.raises(ValueError, match="url must not be empty"):
            extract_web_page("   ")


# ─────────────────────────────────────────────────────────────────────────────
# OneCompiler server tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOneCompilerServer:
    """Verify remote execution request construction without running code locally."""

    @patch("mcp_servers.onecompiler_server.httpx.post")
    def test_rapidapi_uses_supported_execution_endpoint(self, mock_post, monkeypatch):
        monkeypatch.setenv("ONECOMPILER_API_KEY", "test-key")
        monkeypatch.setenv("ONECOMPILER_PROVIDER", "rapidapi")
        response = MagicMock()
        response.json.return_value = {"stdout": "5"}
        response.raise_for_status = MagicMock()
        mock_post.return_value = response

        result = execute_code("python", "print(2 + 3)")

        assert result["stdout"] == "5"
        request = mock_post.call_args
        assert request.args[0] == (
            "https://onecompiler-1.p.rapidapi.com/api/code/exec"
        )
        assert request.kwargs["headers"]["x-rapidapi-host"] == (
            "onecompiler-1.p.rapidapi.com"
        )

    @patch("mcp_servers.onecompiler_server.httpx.post")
    def test_official_key_ignores_stale_rapidapi_provider(
        self, mock_post, monkeypatch
    ):
        monkeypatch.setenv("ONECOMPILER_API_KEY", "oc_test-key")
        monkeypatch.setenv("ONECOMPILER_PROVIDER", "rapidapi")
        response = MagicMock()
        response.json.return_value = {"stdout": "5"}
        response.raise_for_status = MagicMock()
        mock_post.return_value = response

        execute_code("python", "print(2 + 3)")

        request = mock_post.call_args
        assert request.args[0] == "https://onecompiler.com/api/code/exec"
        assert request.kwargs["headers"]["X-API-Key"] == "oc_test-key"

    @patch("mcp_servers.onecompiler_server.httpx.post")
    def test_http_errors_include_provider_response(
        self, mock_post, monkeypatch
    ):
        monkeypatch.setenv("ONECOMPILER_API_KEY", "test-key")
        monkeypatch.setenv("ONECOMPILER_PROVIDER", "direct")
        response = MagicMock()
        response.status_code = 404
        response.text = '{"message":"API does not exist"}'
        mock_post.return_value = response
        from httpx import HTTPStatusError, Request, Response
        mock_post.return_value.raise_for_status.side_effect = HTTPStatusError(
            "not found",
            request=Request("POST", "https://onecompiler.com/api/code/exec"),
            response=Response(404),
        )
        mock_post.return_value.text = '{"message":"API does not exist"}'

        with pytest.raises(RuntimeError, match="HTTP 404"):
            execute_code("python", "print(2 + 3)")


# ─────────────────────────────────────────────────────────────────────────────
# Memory server tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryServer:
    """Tests for memory_set, memory_get, memory_list_keys, memory_delete."""

    def setup_method(self):
        """Use an in-memory Redis-shaped fake without requiring Docker."""
        class FakeRedis:
            def __init__(self):
                self.hashes = {}

            def hset(self, name, key, value):
                self.hashes.setdefault(name, {})[key] = value

            def hget(self, name, key):
                return self.hashes.get(name, {}).get(key)

            def hkeys(self, name):
                return list(self.hashes.get(name, {}))

            def hdel(self, name, key):
                values = self.hashes.get(name, {})
                if key not in values:
                    return 0
                del values[key]
                return 1

            def hgetall(self, name):
                return dict(self.hashes.get(name, {}))

            def delete(self, name):
                return int(self.hashes.pop(name, None) is not None)

        memory_server._redis_client = FakeRedis()

    def teardown_method(self):
        memory_server._redis_client = None

    def test_set_and_get_simple_value(self):
        """Basic round-trip: set a value and get it back."""
        memory_set("session-1", "goal", "Learn Python closures")
        result = memory_get("session-1", "goal")
        assert result == "Learn Python closures"

    def test_get_missing_key_returns_null_string(self):
        """Getting a key that doesn't exist should return 'null', not raise."""
        result = memory_get("session-1", "nonexistent_key")
        assert result == "null"

    def test_get_missing_session_returns_null(self):
        """Getting from a session that doesn't exist should return 'null'."""
        result = memory_get("session-never-created", "any_key")
        assert result == "null"

    def test_sessions_are_isolated(self):
        """Data stored in session-1 should not appear in session-2."""
        memory_set("session-1", "key", "value-for-session-1")
        result = memory_get("session-2", "key")
        assert result == "null"

    def test_overwrite_existing_value(self):
        """Setting the same key twice should update to the new value."""
        memory_set("session-1", "score", "0.6")
        memory_set("session-1", "score", "0.9")
        result = memory_get("session-1", "score")
        assert result == "0.9"

    def test_json_values_round_trip(self):
        """JSON-serialized complex data should survive a round trip."""
        data = {"topics": ["closures", "decorators"], "score": 0.85}
        memory_set("session-1", "progress", json.dumps(data))
        retrieved = memory_get("session-1", "progress")
        parsed = json.loads(retrieved)
        assert parsed["topics"] == ["closures", "decorators"]
        assert parsed["score"] == 0.85

    def test_structured_values_are_serialized_automatically(self):
        """MCP tool calls may provide a list directly instead of a JSON string."""
        memory_set("session-1", "topics", ["Python disclosure"])
        assert memory_get("session-1", "topics") == '["Python disclosure"]'

    def test_list_keys_empty_for_new_session(self):
        """A session with no data should return empty key list."""
        result = memory_list_keys("brand-new-session")
        assert result == []

    def test_list_keys_returns_all_stored_keys(self):
        """list_keys should return all keys stored in the session."""
        memory_set("session-1", "key_a", "value_a")
        memory_set("session-1", "key_b", "value_b")
        memory_set("session-1", "key_c", "value_c")
        keys = memory_list_keys("session-1")
        assert set(keys) == {"key_a", "key_b", "key_c"}

    def test_delete_existing_key(self):
        """Deleting an existing key should make it inaccessible."""
        memory_set("session-1", "temp_key", "temp_value")
        memory_delete("session-1", "temp_key")
        result = memory_get("session-1", "temp_key")
        assert result == "null"

    def test_delete_nonexistent_key_does_not_raise(self):
        """Deleting a key that doesn't exist should return gracefully."""
        result = memory_delete("session-1", "nonexistent")
        assert "not found" in result.lower()

    def test_delete_session_removes_all_memory(self):
        memory_set("session-1", "a", "one")
        memory_set("session-1", "b", "two")
        assert "Deleted all memory" in memory_delete_session("session-1")
        assert memory_list_keys("session-1") == []

    def test_set_returns_confirmation(self):
        """memory_set should return a confirmation message string."""
        result = memory_set("session-1", "key", "value")
        assert isinstance(result, str)
        assert "session-1" in result
        assert "key" in result

    def test_multiple_sessions_independent(self):
        """Multiple sessions should not interfere with each other."""
        for i in range(5):
            memory_set(f"session-{i}", "data", f"value-{i}")
        for i in range(5):
            assert memory_get(f"session-{i}", "data") == f"value-{i}"


class TestGetSessionSummary(TestMemoryServer):
    """Tests for the notes://session/{session_id} resource."""

    def test_empty_session_returns_no_data_message(self):
        result = get_session_summary("empty-session")
        assert "No data stored yet" in result

    def test_populated_session_contains_keys(self):
        memory_set("test-session", "explained_topics", '["closures"]')
        memory_set("test-session", "last_score", "0.85")
        result = get_session_summary("test-session")
        assert "explained_topics" in result
        assert "last_score" in result

    def test_result_is_markdown_formatted(self):
        memory_set("test-session", "any_key", "any_value")
        result = get_session_summary("test-session")
        assert "# Session Memory:" in result
