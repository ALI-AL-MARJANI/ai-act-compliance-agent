"""v2 — the LangGraph agent, with its tools served by an MCP server.

v1: the tools are Python functions IMPORTED into the agent (same process).
v2: the agent starts v2_mcp/server.py as a subprocess, ASKS it which tools
    exist (with their JSON schemas), and calls them over the MCP protocol.
    The agent does not need to know how the tools are implemented.

Why it matters in a company:
  - one tool server, many clients: this agent, Claude Desktop, an IDE, another team's agent;
  - separation of concerns: the compliance team owns the server (rules, knowledge,
    guardrails); agent builders only consume it;
  - governance: one place to update the knowledge, log every call, control access.

What stays in the agent (client side):
  - the LLM and the graph (same as v1);
  - ask_user and the report approval: talking to the HUMAN is the client's job.

Run it (from the repo root):
    python -m v2_mcp.agent "A chatbot that answers questions about our products."
"""

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

# Reused from v0/v1: prompt, model settings, state, routing, human-side functions.
from v0_scratch.agent import LLM_OPTIONS, MODEL, SYSTEM_PROMPT, TOOL_SCHEMAS, wrap_description
from v1_langgraph.agent import (RECURSION_LIMIT, State, _ask_user, _log_update, _short,
                                answer_in_terminal, approve_in_terminal, nudge_node, route_after_agent)

ROOT = Path(__file__).resolve().parent.parent

# How to start the MCP server (stdio = subprocess talking over stdin/stdout).
MCP_SERVERS = {
    "compliance": {
        "command": sys.executable,
        "args": [str(ROOT / "v2_mcp" / "server.py")],
        "transport": "stdio",
    }
}
# ask_user is a LOCAL tool (not on the server): we reuse its schema from v0.
ASK_USER_SCHEMA = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "ask_user")


def build_graph(mcp_tools: list):
    """Same graph as v1, but tool calls go to the MCP server."""
    tools_by_name = {tool.name: tool for tool in mcp_tools}
    # The LLM sees the schemas DISCOVERED from the server + the local ask_user.
    llm = ChatOllama(model=MODEL, **LLM_OPTIONS).bind_tools(mcp_tools + [ASK_USER_SCHEMA])

    async def agent_node(state: State) -> dict:
        response = await llm.ainvoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    async def call_mcp(name: str, args: dict) -> dict:
        """Call a tool on the MCP server and turn its answer back into a dict."""
        if name not in tools_by_name:
            return {"error": f"Unknown tool '{name}'. Available tools: "
                             f"{', '.join(list(tools_by_name) + ['ask_user'])}"}
        output = await tools_by_name[name].ainvoke(args)
        # MCP returns a list of content blocks; ours is one text block with JSON inside.
        text = output[0]["text"] if isinstance(output, list) else str(output)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": text}  # e.g. a validation error raised by the MCP server itself

    async def tools_node(state: State) -> dict:
        questions_asked = state["questions_asked"]
        report_path = state["report_path"]
        tool_messages = []
        for call in state["messages"][-1].tool_calls:
            name, args = call["name"], call["args"]
            if name == "ask_user":  # local: pause the graph, like in v1
                result = _ask_user(args, questions_asked)
                if "answer" in result:
                    questions_asked += 1
            elif name == "save_report":
                # Approval FIRST (client side), then the server checks and writes.
                # Unlike v1, the human may see a draft that the server will refuse:
                # the guardrails live on the server, the agent does not import them.
                decision = interrupt({"type": "approval", **args})
                if decision["action"] == "approve":
                    result = await call_mcp(name, args)
                elif decision["action"] == "edit":
                    result = await call_mcp(name, {**args, "content_markdown": decision["content"]})
                else:
                    result = {"error": f"The human reviewer REJECTED the report. "
                                       f"Feedback: {decision.get('feedback', '')}"}
                report_path = result.get("path", report_path)
            else:
                result = await call_mcp(name, args)
            tool_messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                             tool_call_id=call["id"], name=name))
        return {"messages": tool_messages, "questions_asked": questions_asked, "report_path": report_path}

    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("nudge", nudge_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", "nudge", END])
    builder.add_edge("tools", "agent")
    builder.add_edge("nudge", "agent")
    return builder.compile(checkpointer=InMemorySaver())


async def run_turn(graph, user_text: str, thread_id: str, first_turn: bool,
                   answer_fn=answer_in_terminal, approve_fn=approve_in_terminal, verbose: bool = True) -> dict:
    """Same as v1.run_turn, but async (MCP calls are network-style I/O)."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    content = wrap_description(user_text) if first_turn else user_text
    graph_input = {"messages": [HumanMessage(content)], "questions_asked": 0,
                   "nudges": 0, "report_path": None}
    trace = {"tool_calls": [], "steps": 0, "nudges": 0, "final_answer": None,
             "report": None, "stopped_by": None, "latency_s": None}
    start = time.time()
    try:
        while True:
            interrupted = None
            async for update in graph.astream(graph_input, config, stream_mode="updates"):
                if "__interrupt__" in update:
                    interrupted = update["__interrupt__"][0].value
                    continue
                _log_update(update, trace, verbose)
            if interrupted is None:
                break
            if interrupted["type"] == "question":
                resume_value = answer_fn(interrupted["question"])
            else:
                resume_value = approve_fn(interrupted)
            graph_input = Command(resume=resume_value)
        trace["stopped_by"] = "final_answer"
    except GraphRecursionError:
        trace["stopped_by"] = "max_steps"
        if verbose:
            print(f"\n⛔ Stopped: recursion limit ({RECURSION_LIMIT} node runs) reached.")
    state = (await graph.aget_state(config)).values
    trace["final_answer"] = state["messages"][-1].content
    trace["latency_s"] = round(time.time() - start, 1)
    return trace


async def main() -> None:
    client = MultiServerMCPClient(MCP_SERVERS)
    # Keep ONE session open for the whole conversation (otherwise each tool
    # call would start a new server process).
    async with client.session("compliance") as session:
        mcp_tools = await load_mcp_tools(session)
        print(f"🔌 Connected to MCP server, tools discovered: {[t.name for t in mcp_tools]}")
        graph = build_graph(mcp_tools)

        thread_id = str(uuid.uuid4())
        user_text = " ".join(sys.argv[1:]) or input("Describe the AI system to assess:\n> ")
        first_turn = True
        while user_text.strip():
            trace = await run_turn(graph, user_text, thread_id, first_turn)
            print(f"\n🏁 {_short(trace['final_answer'] or '(no final answer)', 500)}")
            print(f"   steps={trace['steps']}  stopped_by={trace['stopped_by']}  latency={trace['latency_s']}s")
            if trace["report"]:
                print(f"   report saved to {trace['report']['path']}")
            first_turn = False
            user_text = input("\n💬 Follow-up on the same thread (empty = quit): ")


if __name__ == "__main__":
    asyncio.run(main())
