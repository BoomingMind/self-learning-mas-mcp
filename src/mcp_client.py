"""Shared MCP stdio client helpers."""

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def server_config() -> dict[str, dict[str, object]]:
    """Return the project's MCP server subprocess definitions."""
    servers_dir = Path(__file__).resolve().parent / "mcp_servers"
    return {
        name: {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(servers_dir / f"{name}_server.py")],
            "env": dict(os.environ),
        }
        for name in ("memory", "tavily", "onecompiler")
    }


async def call_tool(server_name: str, tool_name: str, arguments: dict) -> object:
    """Call one MCP tool through a real stdio server session."""
    client = MultiServerMCPClient(server_config())
    async with AsyncExitStack() as stack:
        session = await stack.enter_async_context(client.session(server_name))
        tools = await load_mcp_tools(session)
        tool = next((candidate for candidate in tools if candidate.name == tool_name), None)
        if tool is None:
            raise ValueError(f"MCP tool '{tool_name}' was not found on '{server_name}'")
        return await tool.ainvoke(arguments)
