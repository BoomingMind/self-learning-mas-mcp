from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Memory Server")

# In-process store: {session_id: {key: {"value": str, "updated_at": str}}}
# For production: replace with Redis or PostgreSQL.
# The MCP interface stays identical. Only this dict changes.
_store: dict[str, dict] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@mcp.tool()
def memory_set(session_id: str, key: str, value: str) -> str:
    """
    Store a value in session memory.

    Values are always strings. Use JSON for complex data:
    memory_set(session_id, 'quiz_scores', json.dumps([0.8, 0.6]))

    Args:
        session_id: Scopes this data to one study session.
        key: Descriptive name. Examples: 'explained_topics', 'last_quiz_score'
        value: String value. Use JSON for lists or dicts.
    """
    if session_id not in _store:
        _store[session_id] = {}
    _store[session_id][key] = {"value": value, "updated_at": _now_iso()}
    return f"Stored '{key}' for session '{session_id}'"


@mcp.tool()
def memory_get(session_id: str, key: str) -> str:
    """
    Retrieve a value from session memory.

    Returns the stored value, or the string "null" if the key doesn't exist.
    Returns "null" (not Python None) so the LLM can handle the missing case
    without type errors.
    """
    session = _store.get(session_id, {})
    entry = session.get(key)
    return "null" if entry is None else entry["value"]


@mcp.tool()
def memory_list_keys(session_id: str) -> list[str]:
    """List all keys stored for a session. Returns [] if none exist."""
    return list(_store.get(session_id, {}).keys())


@mcp.tool()
def memory_delete(session_id: str, key: str) -> str:
    """Delete a specific key from session memory."""
    session = _store.get(session_id, {})
    if key in session:
        del session[key]
        return f"Deleted '{key}' from session '{session_id}'"
    return f"Key '{key}' not found in session '{session_id}'"


@mcp.resource("notes://session/{session_id}")
def get_session_summary(session_id: str) -> str:
    """Full summary of everything stored for a session. URI: notes://session/{session_id}"""
    session = _store.get(session_id, {})
    if not session:
        return f"# Session Memory: {session_id}\n\nNo data stored yet."
    lines = [f"# Session Memory: {session_id}\n"]
    for key, entry in sorted(session.items()):
        lines.append(f"## {key}")
        lines.append(f"- Value: {entry['value']}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[Memory MCP] Starting server")
    mcp.run()