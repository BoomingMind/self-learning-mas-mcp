import os


def _langfuse_configured() -> bool:
    """
    Check whether Langfuse credentials are present in the environment.

    Returns False if either key is missing or empty. In that case the
    system runs without observability rather than raising an error.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    return bool(public_key and secret_key)

def get_langfuse_handler(session_id: str, user_id: str = "local"):
    """
    Create a Langfuse callback handler for a session, or None if not configured.

    The handler is a LangChain CallbackHandler that Langfuse provides.
    When attached to graph.invoke(), it intercepts every LLM call, tool call,
    and chain invocation automatically. No changes to agent code required.
    """
    if not _langfuse_configured():
        return None

    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            session_id=session_id,
            user_id=user_id,
            tags=["learning-accelerator", "local-inference"],
            metadata={
                "model":     os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
                "framework": "langgraph",
            },
        )
    except ImportError:
        print("[Observability] langfuse not installed. Run: pip install langfuse")
        return None
    except Exception as e:
        print(f"[Observability] Failed to create handler: {e}")
        return None

def get_langfuse_config(
    session_id: str,
    user_id: str = "local",
    extra_config: dict | None = None,
) -> dict:
    """
    Build the complete LangGraph run config for a session.

    Merges the checkpoint thread_id with the Langfuse callback handler.
    This is the only function main.py calls. One function, one config dict,
    everything set up.

    Returns a dict ready to pass as `config` to graph.invoke().
    """
    config = {
        "configurable": {"thread_id": session_id},
    }

    if extra_config:
        config.update(extra_config)

    handler = get_langfuse_handler(session_id, user_id)
    if handler:
        config["callbacks"] = [handler]
        print(f"[Observability] Tracing session {session_id} → "
              f"{os.getenv('LANGFUSE_HOST', 'http://localhost:3000')}")
    else:
        print(f"[Observability] Langfuse not configured. Running without tracing.")

    return config

def flush_langfuse() -> None:
    """
    Flush pending traces before process exit.

    Langfuse sends traces in a background thread. Without this call,
    the last few seconds of traces may be lost when the process exits.
    Call this at the end of main.py, after all graph.invoke() calls.
    """
    if not _langfuse_configured():
        return
    try:
        from langfuse import Langfuse
        Langfuse().flush()
    except Exception:
        pass  # Best-effort. Don't crash on exit.