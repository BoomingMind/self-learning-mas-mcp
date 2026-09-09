import json, os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from graph.state import get_current_topic
from mcp_servers.filesystem_server import list_study_files, read_study_file, search_notes
from mcp_servers.memory_server import memory_get, memory_set

MODEL_NAME = os.getenv("OPEN_OPENROUTER_MODEL", "thinkingmachines/inkling:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


@tool
def tool_list_files() -> list[str]:
    """
    List all available study note files in the notes directory.
    Returns filenames like ['closures.md', 'decorators.md'].
    Call this FIRST to discover what materials exist before reading any file.
    """
    return list_study_files()


@tool
def tool_read_file(filename: str) -> str:
    """
    Read the complete content of a study note file.
    Args:
        filename: Exact filename as returned by tool_list_files().
    Returns the full file text, or an error string if not found.
    """
    return read_study_file(filename)


@tool
def tool_search_notes(query: str) -> str:
    """
    Search across all study notes for a keyword or phrase.
    Args:
        query: Search term (case-insensitive). Example: 'nonlocal', 'closure'
    Returns a JSON string with matching lines and their file locations.
    """
    results = search_notes(query)
    if not results:
        return "No matches found."
    return json.dumps(results, indent=2)


@tool
def tool_memory_get(session_id: str, key: str) -> str:
    """
    Retrieve a value from session memory.
    Args:
        session_id: The current session ID (from state).
        key: The memory key to look up.
    Returns the stored value, or 'null' if not found.
    """
    return memory_get(session_id, key)


@tool
def tool_memory_set(session_id: str, key: str, value: str) -> str:
    """
    Store a value in session memory for later agents to read.
    Args:
        session_id: The current session ID (from state).
        key: Descriptive key name.
        value: String value. Use JSON for complex data.
    """
    return memory_set(session_id, key, value)


EXPLAINER_TOOLS = [
    tool_list_files, tool_read_file, tool_search_notes,
    tool_memory_get, tool_memory_set,
]
TOOL_MAP = {t.name: t for t in EXPLAINER_TOOLS}

EXPLAINER_SYSTEM_PROMPT = """You are an expert tutor explaining topics to a student.

Your explanations must be grounded in the student's actual study materials.
Use the available tools to find and read relevant notes before explaining.

APPROACH (follow this sequence):
1. Call tool_list_files() to see what materials are available
2. Call tool_search_notes(topic) to find which files cover this topic
3. Call tool_read_file(filename) to read the most relevant file(s)
4. Check prior context: call tool_memory_get(session_id, 'explained_topics')
5. Write your explanation based on what you found in the notes

EXPLANATION FORMAT:
- Start with a real-world analogy (1-2 sentences)
- State the core concept clearly (2-3 sentences)
- Show a concrete code example from the student's notes
- End with one common mistake or gotcha to watch out for

After writing the explanation, store what you explained:
  tool_memory_set(session_id, 'explained_topics', <comma-separated topic titles>)
"""

def execute_tool_call(tool_call: dict) -> str:
    """Execute a tool call and return the result as a string. Never raises."""
    name = tool_call["name"]
    args = tool_call["args"]
    if name not in TOOL_MAP:
        return f"Error: unknown tool '{name}'. Available: {list(TOOL_MAP.keys())}"
    try:
        result = TOOL_MAP[name].invoke(args)
        if isinstance(result, (list, dict)):
            return json.dumps(result)
        return str(result)
    except Exception as e:
        return f"Error executing {name}({args}): {type(e).__name__}: {e}"


def explainer_node(state: dict) -> dict:
    """
    LangGraph node: Explainer Agent

    Reads:  state["roadmap"], state["current_topic_index"], state["session_id"]
    Writes: state["messages"], state["error"]
    """
    topic = get_current_topic(state)
    if topic is None:
        return {"error": "No current topic found."}

    session_id = state.get("session_id", "unknown")
    print(f"\n[Explainer] Topic: '{topic.title}'")

    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        temperature=0.1,
    ).bind_tools(EXPLAINER_TOOLS)

    messages = [
        SystemMessage(content=EXPLAINER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Please explain this topic to me: '{topic.title}'\n"
            f"Context: {topic.description}\n"
            f"Session ID for memory calls: {session_id}"
        )),
    ]

    max_iterations = 8
    final_response = None

    for iteration in range(max_iterations):
        print(f"[Explainer] LLM call {iteration + 1}/{max_iterations}...")
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            final_response = response
            print(f"[Explainer] Complete after {iteration + 1} LLM call(s)")
            break

        print(f"[Explainer] {len(response.tool_calls)} tool call(s) requested:")
        for tool_call in response.tool_calls:
            print(f"  → {tool_call['name']}({tool_call['args']})")
            result = execute_tool_call(tool_call)
            log_result = result[:100] + "..." if len(result) > 100 else result
            print(f"    ← {log_result}")

            # The tool_call_id must match the ID the LLM assigned to the request.
            # Without this, the LLM can't correlate result to request.
            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],
            ))

    if final_response is None:
        return {
            "messages": messages,
            "error": f"Explainer reached max iterations ({max_iterations}).",
        }

    print(f"[Explainer] Explanation: {len(final_response.content)} characters")
    return {"messages": messages, "error": None}