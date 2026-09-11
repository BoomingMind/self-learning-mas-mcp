# self-learning-mas-mcp

Self-learning multi-agent system built with LangGraph, LangChain tool calling,
and independent MCP servers.

## Architecture

- LangGraph coordinates the planner, explainer, quiz, and coaching nodes.
- Each LLM-backed node is a LangChain agent created with `create_agent`;
  LangGraph nodes invoke those agents as workflow steps.
- The Explainer binds the LangChain chat model to tools discovered from MCP
  servers; it does not import server functions for production execution.
- The Explainer is an interactive tutor that selectively uses learner memory,
  Tavily web search, and remote OneCompiler execution. Study notes are not
  required by the core workflow.
- The Progress Coach is a native LangGraph node. Only the optional CrewAI
  Study Buddy is exposed through A2A and controlled by `USE_STUDY_BUDDY`.
- `memory_server.py`, `tavily_server.py`, and `onecompiler_server.py` are
  standalone FastMCP processes using JSON-RPC over stdio.
- `MultiServerMCPClient` starts only the active memory, Tavily, and
  OneCompiler servers, keeps their sessions open for the agent turn, and
  exposes their MCP tools as LangChain tools. The legacy filesystem server
  remains available for compatibility tests but is not part of the learning
  workflow.

Install dependencies with `pip install -r requirements.txt`, then run:

```text
python main.py "Learn Python closures from scratch"
```
