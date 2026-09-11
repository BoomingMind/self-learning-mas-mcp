"""Shared model-provider configuration for Ollama, OpenRouter, and Groq."""

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def configured_provider() -> str:
    """Return the default provider, preferring OpenRouter when configured."""
    return os.getenv("MODEL_PROVIDER", "ollama").strip().lower()


def configured_model(provider: str | None = None) -> str:
    provider = (provider or configured_provider()).lower()
    if provider == "openrouter":
        return os.getenv("OPENROUTER_MODEL", "").strip()
    if provider == "groq":
        return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
    return os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip()


def build_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    json_mode: bool = False,
    reasoning_effort: str | None = None,
) -> BaseChatModel:
    """Build the selected chat model without exposing credentials."""
    selected_provider = (provider or configured_provider()).lower()
    selected_model = (model or configured_model(selected_provider)).strip()

    if not selected_model:
        raise ValueError(f"No model configured for provider '{selected_provider}'")

    if selected_provider in {"openrouter", "groq"}:
        env_key = "OPENROUTER_API_KEY" if selected_provider == "openrouter" else "GROQ_API_KEY"
        api_key = os.getenv(env_key, "").strip()
        if not api_key:
            raise RuntimeError(f"{env_key} is not configured")
        kwargs: dict[str, Any] = {
            "model": selected_model,
            "api_key": api_key,
            "temperature": temperature,
        }
        if selected_provider == "openrouter":
            kwargs["base_url"] = os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            )
        else:
            kwargs["base_url"] = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        # OpenRouter models do not consistently support the OpenAI
        # structured-output parameter. Callers already provide JSON-only
        # instructions and validate/normalize the response themselves.
        return ChatOpenAI(**kwargs)

    if selected_provider != "ollama":
        raise ValueError(f"Unsupported model provider: {selected_provider}")

    kwargs = {
        "model": selected_model,
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "temperature": temperature,
    }
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(
        **kwargs,
    )
