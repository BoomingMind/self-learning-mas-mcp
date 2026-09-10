"""
tests/test_explainer.py

Unit tests for the Explainer agent.

These tests validate everything EXCEPT the LLM call:
  - Tool wrapping and dispatch
  - execute_tool_call error handling
  - Tool result formatting

The full integration test (LLM + tools together) requires Ollama.
It is in a separate integration test file (added in Batch 9).

Run: python -m pytest tests/test_explainer.py -v
"""

import json
import pytest

from agents.explainer import (
    execute_tool_call,
    TOOL_MAP,
    EXPLAINER_TOOLS,
)
from graph.state import StudyRoadmap, Topic, initial_state, get_current_topic


class TestToolMap:
    """Verify the tool registry is set up correctly."""

    def test_all_expected_tools_registered(self):
        """All five tools should be in the tool map."""
        expected = {
            "tool_list_files",
            "tool_read_file",
            "tool_search_notes",
            "tool_memory_get",
            "tool_memory_set",
        }
        assert set(TOOL_MAP.keys()) == expected

    def test_tool_map_matches_tools_list(self):
        """TOOL_MAP and EXPLAINER_TOOLS should be consistent."""
        assert len(TOOL_MAP) == len(EXPLAINER_TOOLS)
        for t in EXPLAINER_TOOLS:
            assert t.name in TOOL_MAP

    def test_all_tools_have_descriptions(self):
        """Every tool must have a non-empty description (the LLM reads it)."""
        for t in EXPLAINER_TOOLS:
            assert t.description, f"Tool '{t.name}' has no description"
            assert len(t.description) > 20, (
                f"Tool '{t.name}' description is too short: '{t.description}'"
            )


class TestExecuteToolCall:
    """Tests for the execute_tool_call dispatch function."""

    def test_unknown_tool_returns_error_string(self):
        """An unknown tool name should return an error string, not raise."""
        result = execute_tool_call({
            "name": "nonexistent_tool",
            "args": {},
            "id": "call-001",
        })
        assert "Error" in result
        assert "nonexistent_tool" in result

    def test_tool_list_files_executes(self):
        """tool_list_files should return a JSON list of filenames."""
        result = execute_tool_call({
            "name": "tool_list_files",
            "args": {},
            "id": "call-002",
        })
        # Result is a JSON string of a list
        files = json.loads(result)
        assert isinstance(files, list)
        # Should find at least our sample notes
        assert len(files) >= 1

    def test_tool_read_file_returns_content(self):
        """tool_read_file with a known file should return its content."""
        result = execute_tool_call({
            "name": "tool_read_file",
            "args": {"filename": "closures.md"},
            "id": "call-003",
        })
        assert isinstance(result, str)
        assert len(result) > 0
        assert not result.startswith("Error:")

    def test_tool_read_file_missing_returns_error(self):
        """tool_read_file with a missing file should return error string."""
        result = execute_tool_call({
            "name": "tool_read_file",
            "args": {"filename": "does_not_exist.md"},
            "id": "call-004",
        })
        assert "Error" in result

    def test_tool_search_notes_finds_results(self):
        """tool_search_notes should find content in the sample notes."""
        result = execute_tool_call({
            "name": "tool_search_notes",
            "args": {"query": "closure"},
            "id": "call-005",
        })
        assert isinstance(result, str)
        # Either has results or "No matches", either way, no Error
        assert not result.startswith("Error")

    def test_tool_memory_set_and_get_round_trip(self):
        """Memory set and get should work as a round trip."""
        from mcp_servers.memory_server import _store
        _store.clear()

        set_result = execute_tool_call({
            "name": "tool_memory_set",
            "args": {
                "session_id": "test-session",
                "key": "test_key",
                "value": "test_value",
            },
            "id": "call-006",
        })
        assert "test-session" in set_result

        get_result = execute_tool_call({
            "name": "tool_memory_get",
            "args": {
                "session_id": "test-session",
                "key": "test_key",
            },
            "id": "call-007",
        })
        assert get_result == "test_value"

        _store.clear()

    def test_result_is_always_string(self):
        """execute_tool_call must always return a string (for ToolMessage)."""
        # list_files returns a list, should be JSON-stringified
        result = execute_tool_call({
            "name": "tool_list_files",
            "args": {},
            "id": "call-008",
        })
        assert isinstance(result, str)


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
