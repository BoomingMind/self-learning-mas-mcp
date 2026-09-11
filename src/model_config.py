"""Shared model-provider configuration for Ollama and OpenRouter."""

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
    return os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip()


def build_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> BaseChatModel:
    """Build the selected chat model without exposing credentials."""
    selected_provider = (provider or configured_provider()).lower()
    selected_model = (model or configured_model(selected_provider)).strip()

    if not selected_model:
        raise ValueError(f"No model configured for provider '{selected_provider}'")

    if selected_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        kwargs: dict[str, Any] = {
            "model": selected_model,
            "api_key": api_key,
            "base_url": os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
            "temperature": temperature,
        }
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
