"""
tests/test_curriculum_planner.py

Unit tests for the Curriculum Planner agent.

These tests validate the JSON parsing logic WITHOUT calling Ollama.
Fast, deterministic, and always runnable even without a GPU.

Run: python -m pytest tests/test_curriculum_planner.py -v
"""

import json
import pytest
from agents.curriculum_planner import parse_roadmap_json
from graph.state import StudyRoadmap


class TestParseRoadmapJson:
    """Tests for the parse_roadmap_json function."""

    def _valid_json(self, n_topics=3, include_prereqs=False) -> str:
        """Helper: generate a valid roadmap JSON string."""
        topics = []
        for i in range(n_topics):
            t = {
                "title": f"Topic {i + 1}",
                "description": f"Description for topic {i + 1}",
                "estimated_minutes": 45 + i * 15,
                "prerequisites": [f"Topic {i}"] if include_prereqs and i > 0 else [],
                "status": "pending",
            }
            topics.append(t)

        data = {
            "goal": "Learn Python closures",
            "total_weeks": 2,
            "weekly_hours": 5,
            "topics": topics,
        }
        return json.dumps(data)

    def test_valid_json_returns_study_roadmap(self):
        """A well-formed JSON string should return a StudyRoadmap."""
        result = parse_roadmap_json(self._valid_json())
        assert isinstance(result, StudyRoadmap)

    def test_correct_topic_count(self):
        """The number of topics in the result matches the JSON."""
        result = parse_roadmap_json(self._valid_json(n_topics=5))
        assert len(result.topics) == 5

    def test_goal_is_preserved(self):
        """The goal string from JSON should appear in the roadmap."""
        result = parse_roadmap_json(self._valid_json())
        assert result.goal == "Learn Python closures"

    def test_prerequisites_are_parsed(self):
        """Prerequisites from JSON should appear in Topic objects."""
        result = parse_roadmap_json(self._valid_json(n_topics=3, include_prereqs=True))
        assert result.topics[1].prerequisites == ["Topic 1"]
        assert result.topics[2].prerequisites == ["Topic 2"]

    def test_topic_status_defaults_to_pending(self):
        """All topics should start with status='pending'."""
        result = parse_roadmap_json(self._valid_json())
        for topic in result.topics:
            assert topic.status == "pending"

    def test_invalid_json_raises_value_error(self):
        """Malformed JSON should raise ValueError with a clear message."""
        with pytest.raises(ValueError, match="invalid JSON"):
            parse_roadmap_json("this is { not valid json")

    def test_missing_goal_raises_value_error(self):
        """JSON without 'goal' field should raise ValueError."""
        data = {"total_weeks": 2, "topics": [
            {"title": "T", "description": "D", "estimated_minutes": 30}
        ]}
        with pytest.raises(ValueError, match="goal"):
            parse_roadmap_json(json.dumps(data))

    def test_empty_topics_raises_value_error(self):
        """JSON with empty topics list should raise ValueError."""
        data = {"goal": "Learn X", "total_weeks": 1, "topics": []}
        with pytest.raises(ValueError, match="non-empty"):
            parse_roadmap_json(json.dumps(data))

    def test_topic_missing_required_field_raises_value_error(self):
        """A topic missing 'title' should raise ValueError."""
        data = {
            "goal": "Learn X",
            "total_weeks": 1,
            "topics": [{"description": "D", "estimated_minutes": 30}],  # no title
        }
        with pytest.raises(ValueError, match="title"):
            parse_roadmap_json(json.dumps(data))

    def test_weekly_hours_defaults_when_missing(self):
        """Missing weekly_hours should default to 5."""
        data = {
            "goal": "Learn X",
            "total_weeks": 1,
            "topics": [{"title": "T", "description": "D", "estimated_minutes": 30}],
        }
        result = parse_roadmap_json(json.dumps(data))
        assert result.weekly_hours == 5

    def test_estimated_minutes_is_integer(self):
        """estimated_minutes should be cast to int even if LLM sends a float."""
        data = {
            "goal": "Learn X",
            "total_weeks": 1,
            "topics": [
                {"title": "T", "description": "D", "estimated_minutes": 45.7}
            ],
        }
        result = parse_roadmap_json(json.dumps(data))
        assert isinstance(result.topics[0].estimated_minutes, int)
        assert result.topics[0].estimated_minutes == 45
