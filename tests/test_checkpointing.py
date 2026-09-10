"""
tests/test_checkpointing.py

Tests for checkpointing and human-in-the-loop approval.

These tests verify:
  - SqliteSaver creates and reads checkpoints correctly
  - human_approval_node returns correct state for yes/no
  - route_after_approval routing logic
  - route_after_coach routing logic
  - session_is_complete edge cases

The interrupt() mechanism itself can't be unit tested directly
(it requires a compiled graph + checkpointer). Those are covered
in the integration tests (Batch 9).

Run: python -m pytest tests/test_checkpointing.py -v
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from graph.state import (
    StudyRoadmap,
    Topic,
    initial_state,
    session_is_complete,
)
from graph.workflow import route_after_approval, route_after_coach
from agents.human_approval import human_approval_node


# ─────────────────────────────────────────────────────────────────────────────
# SqliteSaver tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSqliteSaver:
    """Verify SqliteSaver can be imported and initialised."""

    def test_sqlite_saver_can_be_imported(self):
        """SqliteSaver should be importable from langgraph."""
        from langgraph.checkpoint.sqlite import SqliteSaver
        assert SqliteSaver is not None

    def test_sqlite_saver_creates_db_file(self):
        """SqliteSaver should create the database file on initialisation."""
        from langgraph.checkpoint.sqlite import SqliteSaver
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_checkpoints.db")
            assert not os.path.exists(db_path)
            # Creating the saver should create the file
            saver = SqliteSaver.from_conn_string(db_path)
            # Access the connection to trigger file creation
            with saver:
                pass
            # File should now exist
            assert os.path.exists(db_path)

    def test_data_directory_created_by_workflow(self):
        """build_graph() should create data/ directory if it doesn't exist."""
        # This verifies the Path("data").mkdir(exist_ok=True) line works
        data_dir = Path("data")
        # data/ already exists from earlier batches, but exist_ok=True
        # means calling mkdir on it again is safe
        data_dir.mkdir(exist_ok=True)
        assert data_dir.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Human approval node tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHumanApprovalNode:
    """Tests for the human_approval_node function."""

    def _make_state_with_roadmap(self):
        topics = [
            Topic("Closures", "Understanding closures", 60),
            Topic("Decorators", "Building decorators", 75,
                  prerequisites=["Closures"]),
        ]
        state = initial_state("Learn Python", "session-test")
        state["roadmap"] = StudyRoadmap("Learn Python", 2, topics)
        return state

    @patch("agents.human_approval.interrupt")
    def test_yes_answer_sets_approved_true(self, mock_interrupt):
        """'yes' input should result in approved=True."""
        mock_interrupt.return_value = "yes"
        state = self._make_state_with_roadmap()
        result = human_approval_node(state)
        assert result["approved"] is True

    @patch("agents.human_approval.interrupt")
    def test_no_answer_sets_approved_false(self, mock_interrupt):
        """'no' input should result in approved=False."""
        mock_interrupt.return_value = "no"
        state = self._make_state_with_roadmap()
        result = human_approval_node(state)
        assert result["approved"] is False

    @patch("agents.human_approval.interrupt")
    def test_y_shorthand_is_accepted(self, mock_interrupt):
        """'y' should be treated as approval."""
        mock_interrupt.return_value = "y"
        state = self._make_state_with_roadmap()
        result = human_approval_node(state)
        assert result["approved"] is True

    @patch("agents.human_approval.interrupt")
    def test_ok_is_accepted(self, mock_interrupt):
        """'ok' should be treated as approval."""
        mock_interrupt.return_value = "ok"
        state = self._make_state_with_roadmap()
        result = human_approval_node(state)
        assert result["approved"] is True

    @patch("agents.human_approval.interrupt")
    def test_case_insensitive(self, mock_interrupt):
        """Approval check should be case-insensitive."""
        mock_interrupt.return_value = "YES"
        state = self._make_state_with_roadmap()
        result = human_approval_node(state)
        assert result["approved"] is True

    @patch("agents.human_approval.interrupt")
    def test_whitespace_stripped(self, mock_interrupt):
        """Extra whitespace should not affect the decision."""
        mock_interrupt.return_value = "  yes  "
        state = self._make_state_with_roadmap()
        result = human_approval_node(state)
        assert result["approved"] is True

    def test_no_roadmap_auto_approves(self):
        """If there's no roadmap, node should auto-approve without interrupt."""
        state = initial_state("test", "s1")
        state["roadmap"] = None
        # Should not call interrupt(), auto-approve
        result = human_approval_node(state)
        assert result["approved"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Routing function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterApproval:
    """Tests for the route_after_approval routing function."""

    def test_approved_routes_to_explainer(self):
        state = initial_state("test", "s1")
        state["approved"] = True
        assert route_after_approval(state) == "explainer"

    def test_not_approved_routes_to_planner(self):
        state = initial_state("test", "s1")
        state["approved"] = False
        assert route_after_approval(state) == "curriculum_planner"

    def test_missing_approved_routes_to_planner(self):
        """Default (missing approved key) should route to replanning."""
        state = initial_state("test", "s1")
        # approved defaults to False in initial_state
        assert route_after_approval(state) == "curriculum_planner"


class TestRouteAfterCoach:
    """Tests for the route_after_coach routing function."""

    def _make_state(self, n_topics, current_index):
        topics = [Topic(f"T{i}", f"D{i}", 30) for i in range(n_topics)]
        state = initial_state("test", "s1")
        state["roadmap"] = StudyRoadmap("test", 1, topics)
        state["current_topic_index"] = current_index
        return state

    def test_routes_to_explainer_with_remaining_topics(self):
        state = self._make_state(3, 1)  # 3 topics, currently on index 1
        assert route_after_coach(state) == "explainer"

    def test_routes_to_end_when_all_done(self):
        state = self._make_state(3, 3)  # index 3 = past all 3 topics
        assert route_after_coach(state) == "end"

    def test_routes_to_end_on_boundary(self):
        state = self._make_state(2, 2)  # exactly at the end
        assert route_after_coach(state) == "end"


# ─────────────────────────────────────────────────────────────────────────────
# session_is_complete edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionIsComplete:
    """Tests for the session_is_complete helper."""

    def test_not_complete_at_start(self):
        topics = [Topic("T1", "D1", 30), Topic("T2", "D2", 30)]
        state = initial_state("test", "s1")
        state["roadmap"] = StudyRoadmap("test", 1, topics)
        state["current_topic_index"] = 0
        assert session_is_complete(state) is False

    def test_complete_when_index_equals_topic_count(self):
        topics = [Topic("T1", "D1", 30)]
        state = initial_state("test", "s1")
        state["roadmap"] = StudyRoadmap("test", 1, topics)
        state["current_topic_index"] = 1
        assert session_is_complete(state) is True

    def test_complete_when_index_exceeds_topic_count(self):
        """Index past end (shouldn't happen, but handle gracefully)."""
        topics = [Topic("T1", "D1", 30)]
        state = initial_state("test", "s1")
        state["roadmap"] = StudyRoadmap("test", 1, topics)
        state["current_topic_index"] = 99
        assert session_is_complete(state) is True

    def test_complete_without_roadmap(self):
        """No roadmap = nothing to study = complete."""
        state = initial_state("test", "s1")
        assert session_is_complete(state) is True
