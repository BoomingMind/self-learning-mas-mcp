# self-learning-mas-mcp

Self-learning multi-agent system built with LangGraph, LangChain tool calling,
and independent MCP servers.

## Architecture

- LangGraph coordinates the planner, explainer, quiz, and coaching nodes.
- Each LLM-backed node is a LangChain agent created with `create_agent`;
  LangGraph nodes invoke those agents as workflow steps.
- The Explainer binds the LangChain chat model to tools discovered from MCP
  servers; it does not import server functions for production execution.
- `filesystem_server.py` and `memory_server.py` are standalone FastMCP
  processes using JSON-RPC over stdio.
- `MultiServerMCPClient` starts both servers, keeps their sessions open for
  the agent turn, and exposes their MCP tools as LangChain tools.

Install dependencies with `pip install -r requirements.txt`, then run:

```text
python main.py "Learn Python closures from scratch"
```
