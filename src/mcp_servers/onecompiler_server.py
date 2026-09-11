"""Remote OneCompiler MCP tool.

Learner code is sent to OneCompiler and is never executed by this process.
"""

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

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

    provider = os.getenv("ONECOMPILER_PROVIDER", "direct").strip().lower()
    # Official OneCompiler keys use the `oc_` prefix. If an old RapidAPI
    # provider value remains in .env, do not send that key to a dead route.
    if provider == "rapidapi" and api_key.startswith("oc_"):
        provider = "direct"
    headers = {"Content-Type": "application/json"}
    endpoint = os.getenv("ONECOMPILER_ENDPOINT", "").strip() or ONECOMPILER_URL
    if provider == "rapidapi":
        endpoint = RAPIDAPI_URL
        headers.update({
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "onecompiler-1.p.rapidapi.com",
        })
    else:
        headers["X-API-Key"] = api_key

    try:
        response = httpx.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        detail = error.response.text.strip() or "no response body"
        raise RuntimeError(
            f"OneCompiler returned HTTP {error.response.status_code}: {detail}"
        ) from error
    except httpx.HTTPError as error:
        raise RuntimeError(f"OneCompiler request failed: {error}") from error

    try:
        return response.json()
    except ValueError as error:
        raise RuntimeError(
            "OneCompiler returned a non-JSON response"
        ) from error


if __name__ == "__main__":
    mcp.run()
