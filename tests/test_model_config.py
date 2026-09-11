"""Tests for provider selection and per-agent reasoning configuration."""

from model_config import build_chat_model, configured_model


def test_groq_uses_configured_model_and_reasoning_effort(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")

    model = build_chat_model("groq", temperature=0.2, reasoning_effort="medium")

    assert model.model_name == "openai/gpt-oss-120b"
    assert model.reasoning_effort == "medium"


def test_groq_model_defaults_to_requested_model(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert configured_model("groq") == "openai/gpt-oss-120b"
