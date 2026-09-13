"""Integration tests for the real MCP stdio client boundary."""

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp_client import server_config


@pytest.mark.asyncio
async def test_servers_are_discovered_and_called_over_stdio():
    """The client must discover and expose the current MCP servers used by the app."""
    root = Path(__file__).parents[1]
    client = MultiServerMCPClient({
        "memory": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(root / "src" / "mcp_servers" / "memory_server.py")],
            "env": dict(os.environ),
        },
        "tavily": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(root / "src" / "mcp_servers" / "tavily_server.py")],
            "env": dict(os.environ),
        },
    })

    async with AsyncExitStack() as stack:
        memory_session = await stack.enter_async_context(client.session("memory"))
        tavily_session = await stack.enter_async_context(client.session("tavily"))
        memory_tools = await load_mcp_tools(memory_session)
        tavily_tools = await load_mcp_tools(tavily_session)

        memory_tool_names = {t.name for t in memory_tools}
        assert {"memory_set", "memory_get"}.issubset(memory_tool_names)

        tavily_tool_names = {t.name for t in tavily_tools}
        assert {"search_web", "extract_web_page"}.issubset(tavily_tool_names)


def test_rapidapi_onecompiler_uses_hosted_mcp(monkeypatch):
    """RapidAPI credentials select the supplied hosted MCP gateway."""
    monkeypatch.setenv("ONECOMPILER_PROVIDER", "rapidapi")
    monkeypatch.setenv("ONECOMPILER_API_KEY", "test-key")

    config = server_config()["onecompiler"]

    assert config["transport"] == "streamable_http"
    assert config["url"] == "https://mcp.rapidapi.com"
    assert config["headers"]["x-api-host"] == "onecompiler-apis.p.rapidapi.com"
