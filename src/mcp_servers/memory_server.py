"""
mcp_servers/memory_server.py

MCP server providing in-session memory storage.

This server gives agents a shared key-value store scoped to
a session ID. It solves a specific problem: agents in LangGraph
share state via the AgentState TypedDict, but that state only
exists within a single graph invocation.

For data that needs to persist between the Explainer and the
Quiz Generator (within a session) in a queryable format.
rather than just passing through AgentState, this memory
server provides the right interface.

In this project it's used to:
    - Track which topics have been explained (Explainer writes)
    - Store quiz scores per topic (Quiz Generator writes)
    - Provide progress context to the Progress Coach (reads)

Storage:
    Redis provides shared persistent storage. Configure REDIS_URL, or
    REDIS_HOST/REDIS_PORT/REDIS_PASSWORD/REDIS_DB.

Tools exposed:
    memory_set(session_id, key, value)    : store a value
    memory_get(session_id, key)           : retrieve a value
    memory_list_keys(session_id)          : list stored keys
    memory_delete(session_id, key)        : remove a key
    memory_delete_session(session_id)    : remove all session memory

Resources exposed:
    notes://session/{session_id}          : full session summary

Run standalone for testing:
    python mcp_servers/memory_server.py
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, cast
from mcp.server.fastmcp import FastMCP
import redis
from dotenv import load_dotenv

load_dotenv()
mcp = FastMCP("Memory Server")

# ─────────────────────────────────────────────────────────────────────────────
# In-process store
#
# Structure: {session_id: {key: {"value": str, "updated_at": str}}}
#
# session_id scoping means different study sessions cannot
# accidentally read each other's data, important when multiple
# users run the system concurrently.
# ─────────────────────────────────────────────────────────────────────────────

_redis_client: redis.Redis | None = None
REDIS_KEY_PREFIX = "learning-memory:"


def _get_redis_client() -> redis.Redis:
    """Create the shared Redis client lazily for MCP subprocess startup."""
    global _redis_client
    if _redis_client is None:
        url = os.getenv("REDIS_URL", "").strip()
        if url:
            _redis_client = redis.Redis.from_url(url, decode_responses=True)
        else:
            _redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "127.0.0.1"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "1")),
                password=os.getenv("REDIS_PASSWORD", ""),
                decode_responses=True,
            )
    return _redis_client


def _session_key(session_id: str) -> str:
    return f"{REDIS_KEY_PREFIX}{session_id}"


def _now_iso() -> str:
    """Current time as ISO 8601 UTC string."""
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def memory_set(session_id: str, key: str, value: Any) -> str:
    """
    Store a value in session memory.

    Values are stored as strings. Strings are stored unchanged; JSON-compatible
    lists, dictionaries, and scalar values are serialized automatically.
    Example: memory_set(session_id, 'quiz_scores', [0.8, 0.6])

    Args:
        session_id: The current study session ID. Used to scope
                    data so different sessions don't interfere.
        key:        What you're storing. Use descriptive names.
                    Examples: 'explained_topics', 'last_quiz_score',
                    'weak_areas', 'session_start_time'
        value:      The value as a string. Use JSON for complex data.

    Returns:
        Confirmation message with the key and timestamp.
    """
    stored_value = value if isinstance(value, str) else json.dumps(value)

    entry = {
        "value": stored_value,
        "updated_at": _now_iso(),
    }
    _get_redis_client().hset(_session_key(session_id), key, json.dumps(entry))
    return f"Stored '{key}' for session '{session_id}' at {entry['updated_at']}"


@mcp.tool()
def memory_get(session_id: str, key: str) -> str:
    """
    Retrieve a value from session memory.

    Args:
        session_id: The session to look up.
        key:        The key to retrieve.

    Returns:
        The stored value string, or the string "null" if the key
        doesn't exist. Returns "null" (not Python None) so the
        LLM can handle the missing case without type errors.
    """
    raw_entry = _get_redis_client().hget(_session_key(session_id), key)
    if raw_entry is None:
        return "null"
    entry = json.loads(cast(str, raw_entry))
    return str(entry["value"])


@mcp.tool()
def memory_list_keys(session_id: str) -> list[str]:
    """
    List all keys stored for a session.

    Useful for agents that want to check what context is available
    before deciding which memory entries to retrieve.

    Args:
        session_id: The session to inspect.

    Returns:
        List of key names. Empty list if session doesn't exist
        or has no stored data.
    """
    keys = cast(list[str], _get_redis_client().hkeys(_session_key(session_id)))
    return list(keys)


@mcp.tool()
def memory_delete(session_id: str, key: str) -> str:
    """
    Delete a specific key from session memory.

    Useful for cleaning up after a topic is completed or when
    resetting a quiz attempt.

    Args:
        session_id: The session to modify.
        key:        The key to delete.

    Returns:
        Confirmation if deleted, or a message if the key wasn't found.
    """
    if _get_redis_client().hdel(_session_key(session_id), key):
        return f"Deleted '{key}' from session '{session_id}'"
    return f"Key '{key}' not found in session '{session_id}', nothing deleted"


@mcp.tool()
def memory_delete_session(session_id: str) -> str:
    """Delete all Redis learner memory associated with a session."""
    deleted = _get_redis_client().delete(_session_key(session_id))
    if deleted:
        return f"Deleted all memory for session '{session_id}'"
    return f"No memory found for session '{session_id}'"


# ─────────────────────────────────────────────────────────────────────────────
# Resources
# ─────────────────────────────────────────────────────────────────────────────

@mcp.resource("notes://session/{session_id}")
def get_session_summary(session_id: str) -> str:
    """
    Full summary of everything stored for a session.

    Agents can read this resource to get complete context about
    what has happened in the current session, which topics were
    explained, what scores were achieved, what weak areas exist.

    URI: notes://session/{session_id}
    Replace {session_id} with the actual session ID.
    """
    raw_session = cast(
        dict[str, str], _get_redis_client().hgetall(_session_key(session_id))
    )
    session = {key: json.loads(raw) for key, raw in raw_session.items()}
    if not session:
        return f"# Session Memory: {session_id}\n\nNo data stored yet."

    lines = [f"# Session Memory: {session_id}\n"]
    for key, entry in sorted(session.items()):
        lines.append(f"## {key}")
        lines.append(f"- Last updated: {entry['updated_at']}")
        lines.append(f"- Value: {entry['value']}\n")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    # Log startup info to stderr. stdout is the JSON-RPC framing channel
    # under stdio transport, so anything written there would corrupt the protocol.
    print("[Memory MCP] Starting server", file=sys.stderr)
    print("[Memory MCP] Storage: Redis persistent hashes", file=sys.stderr)
    print("[Memory MCP] Transport: stdio", file=sys.stderr)
    print("[Memory MCP] Waiting for connections...", file=sys.stderr)
    mcp.run()
