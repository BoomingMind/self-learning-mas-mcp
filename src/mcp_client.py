"""Shared MCP stdio client helpers."""

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()


def server_config() -> dict[str, dict[str, object]]:
    """Return the project's MCP server subprocess definitions."""
    servers_dir = Path(__file__).resolve().parent / "mcp_servers"
    config: dict[str, dict[str, object]] = {
        name: {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(servers_dir / f"{name}_server.py")],
            "env": dict(os.environ),
        }
        for name in ("memory", "tavily", "onecompiler")
    }
    if os.getenv("ONECOMPILER_PROVIDER", "").strip().lower() == "rapidapi":
        api_key = (
            os.getenv("ONECOMPILER_API_KEY", "").strip()
            or os.getenv("ONE_COMPILER_API_KEY", "").strip()
        )
        if api_key:
            # RapidAPI exposes OneCompiler through its hosted MCP gateway.
            # Keep the key in headers; do not pass it in tool arguments.
            config["onecompiler"] = {
                "transport": "streamable_http",
                "url": "https://mcp.rapidapi.com",
                "headers": {
                    "x-api-host": "onecompiler-apis.p.rapidapi.com",
                    "x-api-key": api_key,
                },
            }
    return config


async def call_tool(server_name: str, tool_name: str, arguments: dict) -> object:
    """Call one MCP tool through a real stdio server session."""
    client = MultiServerMCPClient(server_config())
    async with AsyncExitStack() as stack:
        session = await stack.enter_async_context(client.session(server_name))
        tools = await load_mcp_tools(session)
        tool = next(
            (
                candidate for candidate in tools
                if candidate.name.lower() == tool_name.lower()
            ),
            None,
        )
        if tool is None:
            raise ValueError(f"MCP tool '{tool_name}' was not found on '{server_name}'")
        if (
            server_name == "onecompiler"
            and tool.name.lower() == "execute_code"
            and os.getenv("ONECOMPILER_PROVIDER", "").strip().lower() == "rapidapi"
        ):
            # RapidAPI's hosted MCP tool exposes files instead of the local
            # adapter's code argument.
            arguments = {
                "language": arguments["language"],
                "stdin": arguments.get("stdin", ""),
                "files": [{
                    "name": f"main.{arguments['language']}",
                    "content": arguments["code"],
                }],
            }
        return await tool.ainvoke(arguments)
