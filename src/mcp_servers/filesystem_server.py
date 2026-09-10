"""
mcp_servers/filesystem_server.py

MCP server providing filesystem access to study materials.

This server exposes the student's study notes to any agent
that connects via MCP. It runs as a separate process and
communicates over stdio transport.

Tools exposed:
    list_study_files()         : discover what materials exist
    read_study_file(filename)  : read a specific note file
    search_notes(query)        : find relevant sections by keyword

Resources exposed:
    notes://index              : summary of all available materials

Security:
    Path traversal prevention, agents cannot read files outside
    the designated notes directory.

Run standalone for testing:
    python mcp_servers/filesystem_server.py

Connect from LangGraph agents:
    See agents/explainer.py
"""

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ─────────────────────────────────────────────────────────────────────────────
# Server initialisation
#
# FastMCP("name") creates an MCP server instance.
# The name appears in the Agent Card and in Langfuse traces.
# One line, FastMCP handles all protocol wiring.
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP("Filesystem Server")

# Base path for study materials.
# Read from .env so it can be changed without code modifications.
# The Explainer agent will pass this path when it connects.
NOTES_BASE = Path(os.getenv("NOTES_PATH", "study_materials/sample_notes"))


# ─────────────────────────────────────────────────────────────────────────────
# Tools
#
# @mcp.tool() turns a plain Python function into an MCP Tool.
# The function's:
#   - Name becomes the tool name agents call
#   - Docstring becomes the tool description (the LLM reads this
#     to decide whether to use the tool)
#   - Type annotations become the argument schema
#   - Return type annotation describes what comes back
#
# FastMCP handles serialization, transport, and error propagation.
# You write plain functions.
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_study_files() -> list[str]:
    """
    List all available study note files.

    Returns a list of filenames relative to the notes directory.
    Example: ['closures.md', 'decorators.md', 'python_basics.md']

    Always call this first to discover what materials are available
    before attempting to read specific files.
    """
    if not NOTES_BASE.exists():
        return []

    files = sorted([
        str(f.relative_to(NOTES_BASE))
        for f in NOTES_BASE.rglob("*.md")
    ])
    return files


@mcp.tool()
def read_study_file(filename: str) -> str:
    """
    Read the full content of a study note file.

    Args:
        filename: The filename to read, exactly as returned by
                  list_study_files(). Examples: 'closures.md',
                  'python/variables.md'

    Returns:
        The full text content of the file.
        Returns an error string if the file doesn't exist or
        the path is invalid, never raises an exception, so the
        agent can handle the error gracefully.
    """
    file_path = NOTES_BASE / filename

    # ── Security: path traversal prevention ──────────────────────────
    # Without this check, an agent could call:
    #   read_study_file("../../.env")
    # and read your API keys. We resolve both paths and verify
    # the requested file is inside the notes directory.
    try:
        resolved = file_path.resolve()
        resolved.relative_to(NOTES_BASE.resolve())
    except ValueError:
        return (
            f"Error: path traversal attempt blocked for '{filename}'. "
            "Only files within the notes directory are accessible."
        )

    if not file_path.exists():
        available = list_study_files()
        return (
            f"Error: '{filename}' not found. "
            f"Available files: {available}"
        )

    if file_path.suffix != ".md":
        return f"Error: only .md files are accessible, got '{file_path.suffix}'"

    try:
        content = file_path.read_text(encoding="utf-8")
        return content
    except (PermissionError, OSError) as e:
        return f"Error reading '{filename}': {e}"


@mcp.tool()
def search_notes(query: str) -> list[dict]:
    """
    Search across all study notes for a keyword or phrase.

    Performs case-insensitive substring search across all .md files.
    Returns matching lines with their file and line number context.

    Args:
        query: The search term. Case-insensitive.
               Examples: 'closure', 'nonlocal', 'def make_'

    Returns:
        List of matches, each with keys:
            'file':        relative filename
            'line_number': 1-based line number
            'line':        the matching line text (stripped)
        Maximum 20 results to avoid overwhelming the context window.
        Empty list if no matches found.
    """
    if not NOTES_BASE.exists():
        return []

    results = []
    query_lower = query.lower()

    # Search each file in sorted order for deterministic results
    for file_path in sorted(NOTES_BASE.rglob("*.md")):
        rel_path = str(file_path.relative_to(NOTES_BASE))
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, PermissionError, OSError):
            continue   # Skip unreadable files silently

        for line_num, line in enumerate(lines, 1):
            if query_lower in line.lower():
                results.append({
                    "file": rel_path,
                    "line_number": line_num,
                    "line": line.strip(),
                })
                # Hard cap to prevent context window overflow
                if len(results) >= 20:
                    return results

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Resources
#
# @mcp.resource("uri_pattern") turns a function into an MCP Resource.
# The URI is how agents identify the resource, like a URL.
# Resources are read-only. Agents cannot write to them.
# ─────────────────────────────────────────────────────────────────────────────

@mcp.resource("notes://index")
def get_notes_index() -> str:
    """
    Index of all available study materials.

    Returns a formatted Markdown summary showing all files
    and their sizes. Agents can read this resource to get
    an overview without loading every file.

    URI: notes://index
    """
    files = list_study_files()
    if not files:
        return "# Study Materials Index\n\nNo study materials found."

    lines = ["# Study Materials Index\n"]
    for filename in files:
        file_path = NOTES_BASE / filename
        try:
            size_kb = file_path.stat().st_size / 1024
            lines.append(f"- **{filename}** ({size_kb:.1f} KB)")
        except OSError:
            lines.append(f"- **{filename}** (size unknown)")

    lines.append(f"\nTotal: {len(files)} file(s)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
#
# When run as a script, the server starts in stdio mode.
# LangGraph agents connect to this via subprocess + pipes.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    notes_path_str = str(NOTES_BASE.resolve())
    # Log startup info to stderr. stdout is the JSON-RPC framing channel
    # under stdio transport, so anything written there would corrupt the protocol.
    print("[Filesystem MCP] Starting server", file=sys.stderr)
    print(f"[Filesystem MCP] Serving files from: {notes_path_str}", file=sys.stderr)
    print("[Filesystem MCP] Transport: stdio", file=sys.stderr)
    print("[Filesystem MCP] Waiting for connections...", file=sys.stderr)
    mcp.run()
