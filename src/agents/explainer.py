"""
src/agents/explainer.py

The Explainer agent.

Given a topic from the roadmap, this agent uses MCP tools to retrieve relevant
study material and session context, then produces a clear, personalized
explanation.

Study material is optional supporting context, not the only source of truth.
The agent can explain topics from general knowledge when notes do not cover
them, while clearly distinguishing notes from its own explanation.

Integration note:
  MCP tools are discovered from independent stdio server processes via
  MultiServerMCPClient. The server sessions remain open for the complete
  agent turn so stateful memory calls share one process.
"""

import asyncio
import os
import traceback
from contextlib import AsyncExitStack

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama

from graph.state import get_current_topic
from mcp_client import server_config

# Model configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
#
# Instructs the agent on its role and output contract while leaving tool
# selection to LangChain's tool-calling loop.
# ─────────────────────────────────────────────────────────────────────────────

EXPLAINER_SYSTEM_PROMPT = """You are an expert tutor explaining topics to a student.

Use filesystem tools to find relevant study materials when they can personalize
the explanation, and use session memory when prior explanations provide useful
context. Decide which tools are necessary based on the topic; do not call them
mechanically. Treat notes as supporting context, not as a requirement or
replacement for your general knowledge.

EXPLANATION FORMAT:
- Start with a real-world analogy (1-2 sentences)
- State the core concept clearly (2-3 sentences)
- Show a concrete code example; prefer the student's notes when relevant
- End with one "common mistake" or "gotcha" to watch out for
- Include a short "Quick check" question the student can answer
- Target length: 300-500 words

Use prior session context to adapt the depth and examples when it is useful.
After producing the explanation, record the topic in session memory when the
memory tools are available. Do not claim that a detail came from the notes
unless you actually retrieved it.

If the notes do not cover the topic, explain it from general knowledge and say:
"Your notes don't cover this specifically, but here's the concept:"
"""


def _exception_summary(error: BaseException) -> str:
    """Return the actionable leaf message from nested async task errors."""
    nested = getattr(error, "exceptions", None)
    if nested:
        return "; ".join(_exception_summary(item) for item in nested)
    return str(error)


async def _run_explainer_agent(
    topic_title: str,
    topic_description: str,
    session_id: str,
) -> tuple[list, AIMessage]:
    """Run a LangChain tool-calling agent against persistent MCP sessions."""
    client = MultiServerMCPClient(server_config())
    async with AsyncExitStack() as stack:
        filesystem_session = await stack.enter_async_context(
            client.session("filesystem")
        )
        memory_session = await stack.enter_async_context(client.session("memory"))
        mcp_tools = [
            *await load_mcp_tools(filesystem_session),
            *await load_mcp_tools(memory_session),
        ]
        llm = ChatOllama(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
        )
        agent = create_agent(
            model=llm,
            tools=mcp_tools,
            system_prompt=EXPLAINER_SYSTEM_PROMPT,
            # The outer LangGraph owns PostgreSQL checkpointing. Do not let
            # this nested async agent inherit its synchronous saver.
            checkpointer=False,
            name="explainer",
        )
        result = await agent.ainvoke({
            "messages": [
                HumanMessage(content=(
                    f"Please explain this topic to me: '{topic_title}'\n"
                    f"Context: {topic_description}\n"
                    f"Session ID for memory calls: {session_id}"
                )),
            ],
        })
        messages = result["messages"]
        response = messages[-1]
        print("[Explainer] LangChain agent completed")
        return messages, response


# ─────────────────────────────────────────────────────────────────────────────
# The LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def explainer_node(state: dict) -> dict:
    """
    LangGraph node: Explainer Agent

    Reads:
        state["roadmap"]              : to find the current topic
        state["current_topic_index"]  : which topic to explain
        state["session_id"]           : for memory tool calls

    Writes:
        state["messages"]             : conversation + tool call history
        state["error"]                : error string on failure

    The LangChain agent manages tool calls and returns the final explanation
    with its message history.
    """
    # ── Get current topic ─────────────────────────────────────────────
    topic = get_current_topic(state)
    if topic is None:
        return {
            "error": "No current topic found. Curriculum Planner must run first.",
        }

    session_id = state.get("session_id", "unknown")
    print(f"\n[Explainer] Topic: '{topic.title}'")
    print(f"[Explainer] Description: {topic.description}")

    try:
        messages, final_response = asyncio.run(
            _run_explainer_agent(topic.title, topic.description, session_id)
        )
    except Exception as e:
        detail = _exception_summary(e)
        print(f"[Explainer] Agent failed: {detail}")
        traceback.print_exception(e)
        return {
            "error": f"Explainer agent failed: {detail}",
        }

    explanation_length = len(final_response.content)
    print(f"[Explainer] Explanation: {explanation_length} characters")

    return {
        "messages": messages,
        "error": None,
        "roadmap": state.get("roadmap"),
        "current_topic_index": state.get("current_topic_index", 0),
        "session_id": state.get("session_id", ""),
    }
