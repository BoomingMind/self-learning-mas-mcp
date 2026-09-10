"""
tests/test_observability.py

Tests for the Langfuse observability setup.

These tests run WITHOUT Docker or a Langfuse server.
They validate:
  - Config is built correctly with and without credentials
  - Handler returns None gracefully when credentials missing
  - get_langfuse_config returns correct structure
  - flush_langfuse is a no-op when not configured

Run: python -m pytest tests/test_observability.py -v
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from observability.langfuse_setup import (
    get_langfuse_config,
    get_langfuse_handler,
    flush_langfuse,
    _langfuse_configured,
)


# ─────────────────────────────────────────────────────────────────────────────
# _langfuse_configured tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLangfuseConfigured:
    """Tests for the credential check helper."""

    def test_returns_false_when_keys_missing(self):
        """Missing env vars should return False."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove both keys if present
            env = {k: v for k, v in os.environ.items()
                   if k not in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")}
            with patch.dict(os.environ, env, clear=True):
                assert _langfuse_configured() is False

    def test_returns_false_when_keys_empty(self):
        """Empty string keys should return False."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        }):
            assert _langfuse_configured() is False

    def test_returns_false_when_only_public_key(self):
        """Having only the public key should return False."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            "LANGFUSE_SECRET_KEY": "",
        }):
            assert _langfuse_configured() is False

    def test_returns_true_when_both_keys_present(self):
        """Both keys present should return True."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            "LANGFUSE_SECRET_KEY": "sk-lf-test",
        }):
            assert _langfuse_configured() is True

    def test_strips_whitespace_from_keys(self):
        """Keys with only whitespace should return False."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "   ",
            "LANGFUSE_SECRET_KEY": "   ",
        }):
            assert _langfuse_configured() is False


# ─────────────────────────────────────────────────────────────────────────────
# get_langfuse_handler tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLangfuseHandler:
    """Tests for the callback handler factory."""

    def test_returns_none_when_not_configured(self):
        """No credentials should return None, not raise."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        }):
            handler = get_langfuse_handler("session-001")
            assert handler is None

    def test_returns_none_on_import_error(self):
        """ImportError on the langfuse package should return None gracefully."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            "LANGFUSE_SECRET_KEY": "sk-lf-test",
        }):
            with patch("observability.langfuse_setup._langfuse_configured",
                       return_value=True):
                # Patch the import that get_langfuse_handler does internally
                real_import = __import__

                def fake_import(name, *args, **kwargs):
                    if name.startswith("langfuse"):
                        raise ImportError("langfuse not installed")
                    return real_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=fake_import):
                    handler = get_langfuse_handler("session-001")
                    assert handler is None, (
                        "get_langfuse_handler should swallow ImportError "
                        "and return None"
                    )

    def test_returns_handler_when_configured(self):
        """With valid credentials and langfuse.langchain available, should return a handler."""
        try:
            from langfuse.langchain import CallbackHandler  # noqa: F401
        except ImportError:
            pytest.skip(
                "langfuse.langchain.CallbackHandler not importable. "
                "Skipping handler creation test."
            )

        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            "LANGFUSE_SECRET_KEY": "sk-lf-test",
            "LANGFUSE_HOST": "http://localhost:3000",
        }):
            with patch("observability.langfuse_setup._langfuse_configured",
                       return_value=True):
                handler = get_langfuse_handler("session-001", user_id="test-user")
                assert handler is not None, (
                    "Handler should be created when credentials are present"
                )


# ─────────────────────────────────────────────────────────────────────────────
# get_langfuse_config tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLangfuseConfig:
    """Tests for the main config builder function."""

    def test_always_includes_thread_id(self):
        """Config must always have thread_id for checkpointing."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        }):
            config = get_langfuse_config("test-session-123")
            assert "configurable" in config
            assert config["configurable"]["thread_id"] == "test-session-123"

    def test_no_callbacks_when_not_configured(self):
        """Without credentials, config should have no callbacks key."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        }):
            config = get_langfuse_config("session-001")
            assert "callbacks" not in config

    def test_session_id_in_thread_id(self):
        """The session_id should become the thread_id."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        }):
            config = get_langfuse_config("my-unique-session")
            assert config["configurable"]["thread_id"] == "my-unique-session"

    def test_merges_extra_config(self):
        """Extra config dict should be merged into the result."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        }):
            config = get_langfuse_config(
                "session-001",
                extra_config={"run_name": "test_run", "tags": ["test"]},
            )
            assert config.get("run_name") == "test_run"
            assert config.get("tags") == ["test"]

    def test_returns_dict(self):
        """Should always return a dict."""
        config = get_langfuse_config("any-session")
        assert isinstance(config, dict)

    def test_different_sessions_get_different_thread_ids(self):
        """Two calls with different session IDs should produce distinct configs."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        }):
            config1 = get_langfuse_config("session-A")
            config2 = get_langfuse_config("session-B")
            tid1 = config1["configurable"]["thread_id"]
            tid2 = config2["configurable"]["thread_id"]
            assert tid1 != tid2


# ─────────────────────────────────────────────────────────────────────────────
# flush_langfuse tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFlushLangfuse:
    """Tests for the flush function."""

    def test_noop_when_not_configured(self):
        """flush_langfuse should not raise when Langfuse is not configured."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        }):
            try:
                flush_langfuse()
            except Exception as e:
                pytest.fail(f"flush_langfuse raised {e} when not configured")

    def test_noop_when_langfuse_not_installed(self):
        """Should not raise if langfuse package is missing."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            "LANGFUSE_SECRET_KEY": "sk-lf-test",
        }):
            with patch("observability.langfuse_setup._langfuse_configured",
                       return_value=True):
                with patch("observability.langfuse_setup.Langfuse",
                           side_effect=Exception("connection refused"),
                           create=True):
                    try:
                        flush_langfuse()
                    except Exception as e:
                        pytest.fail(f"flush_langfuse raised {e}")
