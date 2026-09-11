"""Remote OneCompiler MCP tool.

Learner code is sent to OneCompiler and is never executed by this process.
"""

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("OneCompiler Remote Execution")
ONECOMPILER_URL = "https://onecompiler.com/api/code/exec"
RAPIDAPI_URL = "https://onecompiler-1.p.rapidapi.com/api/code/exec"


@mcp.tool()
def execute_code(
    language: str,
    code: str,
    stdin: str = "",
    version: str | None = None,
) -> dict[str, Any]:
    """Execute a code sample remotely through OneCompiler."""
    api_key = os.getenv("ONECOMPILER_API_KEY") or os.getenv("ONE_COMPILER_API_KEY")
    if not api_key:
        raise RuntimeError("OneCompiler API key is not configured")
    if not code.strip():
        raise ValueError("code must not be empty")

    payload: dict[str, Any] = {
        "language": language,
        "stdin": stdin,
        "files": [{"name": "main", "content": code}],
    }
    if version:
        payload["version"] = version

    provider = os.getenv("ONECOMPILER_PROVIDER", "direct").lower()
    headers = {"Content-Type": "application/json"}
    endpoint = ONECOMPILER_URL
    if provider == "rapidapi":
        endpoint = RAPIDAPI_URL
        headers.update({
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "onecompiler-1.p.rapidapi.com",
        })
    else:
        headers["X-API-Key"] = api_key

    response = httpx.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    mcp.run()
