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
    """The client must discover and call tools in separate server processes."""
    root = Path(__file__).parents[1]
    client = MultiServerMCPClient({
        "filesystem": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(root / "src" / "mcp_servers" / "filesystem_server.py")],
            "env": dict(os.environ),
        },
        "memory": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(root / "src" / "mcp_servers" / "memory_server.py")],
            "env": dict(os.environ),
        },
    })

    async with AsyncExitStack() as stack:
        filesystem_session = await stack.enter_async_context(
            client.session("filesystem")
        )
        memory_session = await stack.enter_async_context(client.session("memory"))
        filesystem_tools = await load_mcp_tools(filesystem_session)
        memory_tools = await load_mcp_tools(memory_session)

        list_files = next(t for t in filesystem_tools if t.name == "list_study_files")
        assert "closures.md" in await list_files.ainvoke({})

        memory_set = next(t for t in memory_tools if t.name == "memory_set")
        memory_get = next(t for t in memory_tools if t.name == "memory_get")
        await memory_set.ainvoke({
            "session_id": "stdio-test",
            "key": "topic",
            "value": "closures",
        })
        assert await memory_get.ainvoke({
            "session_id": "stdio-test",
            "key": "topic",
        }) == "closures"


def test_rapidapi_onecompiler_uses_hosted_mcp(monkeypatch):
    """RapidAPI credentials select the supplied hosted MCP gateway."""
    monkeypatch.setenv("ONECOMPILER_PROVIDER", "rapidapi")
    monkeypatch.setenv("ONECOMPILER_API_KEY", "test-key")

    config = server_config()["onecompiler"]

    assert config["transport"] == "streamable_http"
    assert config["url"] == "https://mcp.rapidapi.com"
    assert config["headers"]["x-api-host"] == "onecompiler-apis.p.rapidapi.com"
