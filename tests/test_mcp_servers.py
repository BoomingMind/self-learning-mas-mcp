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

import pytest

# We import the functions directly, not the mcp server object
from mcp_servers.filesystem_server import (
    list_study_files,
    read_study_file,
    search_notes,
    get_notes_index,
    NOTES_BASE,
)
from mcp_servers.memory_server import (
    memory_set,
    memory_get,
    memory_list_keys,
    memory_delete,
    get_session_summary,
    _store,   # direct access for test cleanup
)


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem server tests
# ─────────────────────────────────────────────────────────────────────────────

class TestListStudyFiles:
    """Tests for the list_study_files tool."""

    def test_returns_list(self):
        """Should always return a list, even if empty."""
        result = list_study_files()
        assert isinstance(result, list)

    def test_finds_sample_notes(self):
        """Should find the three sample note files from Batch 1."""
        result = list_study_files()
        # We know these exist from Batch 1 setup
        assert len(result) >= 1, (
            "No .md files found. Make sure study_materials/sample_notes/ "
            "contains the files created in Batch 1."
        )

    def test_returns_only_md_files(self):
        """All returned files should end in .md."""
        result = list_study_files()
        for filename in result:
            assert filename.endswith(".md"), (
                f"Non-.md file returned: {filename}"
            )

    def test_results_are_sorted(self):
        """Results should be in alphabetical order."""
        result = list_study_files()
        assert result == sorted(result)

    def test_returns_relative_paths(self):
        """Paths should be relative, not absolute."""
        result = list_study_files()
        for filename in result:
            assert not filename.startswith("/"), (
                f"Absolute path returned: {filename}"
            )


class TestReadStudyFile:
    """Tests for the read_study_file tool."""

    def test_reads_existing_file(self):
        """Should return file content for a known file."""
        files = list_study_files()
        if not files:
            pytest.skip("No study files available")
        content = read_study_file(files[0])
        assert isinstance(content, str)
        assert len(content) > 0
        assert not content.startswith("Error:")

    def test_closures_file_contains_expected_content(self):
        """The closures.md file should contain closure-related content."""
        content = read_study_file("closures.md")
        assert "closure" in content.lower(), (
            "closures.md doesn't contain 'closure', check the file content"
        )

    def test_nonexistent_file_returns_error_string(self):
        """Missing files should return error string, not raise exception."""
        result = read_study_file("does_not_exist.md")
        assert result.startswith("Error:")
        assert "not found" in result

    def test_path_traversal_blocked(self):
        """Path traversal attempts should return error string."""
        result = read_study_file("../../.env")
        assert result.startswith("Error:")
        assert "traversal" in result.lower()

    def test_non_md_file_blocked(self):
        """Non-.md files should return error string."""
        result = read_study_file("requirements.txt")
        assert result.startswith("Error:")

    def test_returns_string_not_bytes(self):
        """Content should be decoded string, not bytes."""
        files = list_study_files()
        if not files:
            pytest.skip("No study files available")
        content = read_study_file(files[0])
        assert isinstance(content, str)


class TestSearchNotes:
    """Tests for the search_notes tool."""

    def test_returns_list(self):
        """Should always return a list."""
        result = search_notes("python")
        assert isinstance(result, list)

    def test_finds_known_term(self):
        """Searching for 'closure' should find results in closures.md."""
        results = search_notes("closure")
        assert len(results) > 0, (
            "No results for 'closure', check closures.md exists and contains this term"
        )

    def test_result_has_required_keys(self):
        """Each result should have file, line_number, and line keys."""
        results = search_notes("def")
        if not results:
            pytest.skip("No results found for 'def'")
        for result in results:
            assert "file" in result
            assert "line_number" in result
            assert "line" in result

    def test_line_numbers_are_positive_integers(self):
        """Line numbers should be 1-based positive integers."""
        results = search_notes("python")
        for result in results:
            assert isinstance(result["line_number"], int)
            assert result["line_number"] >= 1

    def test_case_insensitive_search(self):
        """Search should be case-insensitive."""
        upper = search_notes("CLOSURE")
        lower = search_notes("closure")
        mixed = search_notes("Closure")
        # All should return the same number of results
        assert len(upper) == len(lower) == len(mixed)

    def test_max_results_is_20(self):
        """Search should return at most 20 results."""
        # Search for 'e', will match many lines
        results = search_notes("e")
        assert len(results) <= 20

    def test_no_match_returns_empty_list(self):
        """Searching for gibberish should return empty list, not error."""
        results = search_notes("xyzzy_impossible_string_12345")
        assert results == []


class TestGetNotesIndex:
    """Tests for the notes://index resource."""

    def test_returns_string(self):
        result = get_notes_index()
        assert isinstance(result, str)

    def test_contains_markdown_header(self):
        result = get_notes_index()
        assert "# Study Materials Index" in result

    def test_lists_known_files(self):
        result = get_notes_index()
        # closures.md was created in Batch 1
        assert "closures.md" in result


# ─────────────────────────────────────────────────────────────────────────────
# Memory server tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryServer:
    """Tests for memory_set, memory_get, memory_list_keys, memory_delete."""

    def setup_method(self):
        """Clear the store before each test for isolation."""
        _store.clear()

    def teardown_method(self):
        """Clear the store after each test."""
        _store.clear()

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


class TestGetSessionSummary:
    """Tests for the notes://session/{session_id} resource."""

    def setup_method(self):
        _store.clear()

    def teardown_method(self):
        _store.clear()

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
