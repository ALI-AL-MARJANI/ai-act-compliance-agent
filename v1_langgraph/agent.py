"""v1 — the same agent, rebuilt with LangGraph.

What LangGraph brings compared to v0 (our hand-written while-loop):
  + EXPLICIT CONTROL FLOW: the agent is a graph (nodes = steps, edges = what
    comes next). The loop "LLM -> tools -> LLM" becomes visible and can be drawn.
  + PERSISTENCE: a checkpointer saves the state after every node. With a
    thread_id we can continue the same conversation later ("what if a human
    always reviews the ranking?") and the agent re-assesses with full context.
  + PAUSE / RESUME: interrupt() stops the graph, saves its state, and returns
    control to us. We resume it later with Command(resume=answer). No blocking
    input() inside the agent: the question could be answered from a web UI,
    a Slack message, or tomorrow morning.
What it costs:
  - a dependency (and its API changes quickly), and an abstraction layer:
    more concepts to learn (state, reducers, checkpointer, Command) and errors
    that are harder to debug than in 100 lines of plain Python.

The graph:
    START -> agent --(tool calls?)--> tools -> agent ...
                   --(stopped too early?)--> nudge -> agent
                   --(done)--> END

Run it (from the repo root):
    python -m v1_langgraph.agent "A recruitment tool that ranks CVs..."
    python -m v1_langgraph.agent --mermaid      # print the graph as a Mermaid diagram
"""

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from tools import compliance_tools as ct
from v0_scratch.agent import LLM_OPTIONS, MAX_NUDGES, MAX_STEPS, MODEL, SYSTEM_PROMPT, TOOL_SCHEMAS, wrap_description

# Same model and same tool schemas as v0: only the "engine" around them changes.
llm = ChatOllama(model=MODEL, **LLM_OPTIONS).bind_tools(TOOL_SCHEMAS)


# ---------------------------------------------------------------------------
# 1. The STATE: everything the graph remembers. The checkpointer saves it
#    after each node, per thread_id.
# ---------------------------------------------------------------------------
class State(TypedDict):
    # add_messages is a "reducer": nodes return NEW messages, and LangGraph
    # appends them to the list (instead of replacing the list).
    messages: Annotated[list[AnyMessage], add_messages]
    questions_asked: int     # for the ask_user limit (v0 used a global variable)
    nudges: int
    report_path: str | None  # set when a report is saved


# ---------------------------------------------------------------------------
# 2. NODES: plain functions that take the state and return an update.
# ---------------------------------------------------------------------------
def agent_node(state: State) -> dict:
    """Call the LLM with the system prompt + the conversation so far."""
    response = llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


def tools_node(state: State) -> dict:
    """Run the tool calls of the last LLM message (same logic as execute_tool in v0).

    Two calls are special and PAUSE the graph with interrupt():
      - ask_user:    we need an answer from the human
      - save_report: we need the human's approval (approve / edit / reject)
    Important: on resume, LangGraph re-runs this whole node from the start, and
    interrupt() then returns the human's answer instead of pausing.
    """
    questions_asked = state["questions_asked"]
    report_path = state["report_path"]
    tool_messages = []

    for call in state["messages"][-1].tool_calls:
        name, args = call["name"], call["args"]

        if name == "ask_user":
            result = _ask_user(args, questions_asked)
            if "answer" in result:
                questions_asked += 1
        elif name == "save_report":
            result = _save_report_with_approval(args)
            report_path = result.get("path", report_path)
        elif name in ct.TOOLS:
            function, _ = ct.TOOLS[name]
            try:
                result = function(**args)
            except TypeError as e:
                result = {"error": f"Invalid arguments for {name}: {e}"}
        else:
            result = {"error": f"Unknown tool '{name}'. Available tools: {', '.join(ct.TOOLS)}"}

        tool_messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                         tool_call_id=call["id"], name=name))

    return {"messages": tool_messages, "questions_asked": questions_asked, "report_path": report_path}


def _ask_user(args: dict, questions_asked: int) -> dict:
    try:
        question = ct.AskUserArgs(**args).question
    except ValidationError as e:
        return {"error": f"Invalid arguments for ask_user: {e.errors()[0]['msg']}"}
    if questions_asked >= ct.MAX_QUESTIONS:
        return {"error": f"Question limit reached ({ct.MAX_QUESTIONS}). Continue with the "
                         "information you have and state your assumptions in the report."}
    answer = interrupt({"type": "question", "question": question})  # <-- PAUSE here
    return {"question": question, "answer": answer}


def _save_report_with_approval(args: dict) -> dict:
    try:
        error = ct.validate_report(**args)  # automatic checks BEFORE bothering the human
    except TypeError as e:
        return {"error": f"Invalid arguments for save_report: {e}"}
    if error:
        return error

    decision = interrupt({"type": "approval", **args})  # <-- PAUSE here
    if decision["action"] == "approve":
        return ct.save_report(**args)
    if decision["action"] == "edit":
        # The human edited the Markdown: save their version (it is re-checked by save_report).
        return ct.save_report(args["title"], args["risk_category"], decision["content"])
    return {"error": f"The human reviewer REJECTED the report. Feedback: {decision.get('feedback', '')}"}


def nudge_node(state: State) -> dict:
    return {"nudges": state["nudges"] + 1,
            "messages": [HumanMessage("You have not saved the report yet. "
                                      "Finish the procedure and call save_report.")]}


# ---------------------------------------------------------------------------
# 3. CONDITIONAL EDGE: after the agent node, where do we go?
# ---------------------------------------------------------------------------
def route_after_agent(state: State) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    if state["report_path"] is None and state["nudges"] < MAX_NUDGES:
        return "nudge"
    return END


# ---------------------------------------------------------------------------
# 4. Build the graph.
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("nudge", nudge_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", "nudge", END])
    builder.add_edge("tools", "agent")
    builder.add_edge("nudge", "agent")
    # The checkpointer saves the state after every node -> pause/resume and memory.
    # InMemorySaver = in RAM only (lost when the program stops); a SqliteSaver or
    # PostgresSaver would keep it on disk.
    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()

# MAX_STEPS in LangGraph = "recursion_limit": max number of node runs per call.
# One agent step = agent node + tools node = 2 runs, plus the nudges.
RECURSION_LIMIT = 2 * MAX_STEPS + 2 * MAX_NUDGES


# ---------------------------------------------------------------------------
# 5. Human side: how the terminal answers the two kinds of interrupts.
# ---------------------------------------------------------------------------
def answer_in_terminal(question: str) -> str:
    return input(f"\n🤖 Agent asks: {question}\n👤 Your answer: ")


def approve_in_terminal(draft: dict) -> dict:
    print("\n" + "=" * 70)
    print(f"📄 DRAFT REPORT — {draft['title']} [{draft['risk_category']}]")
    print("=" * 70)
    print(draft["content_markdown"])
    print("=" * 70)
    choice = input("Approve, edit or reject? (a/e/r): ").strip().lower()
    if choice == "a":
        return {"action": "approve"}
    if choice == "e":
        # Editing Markdown in a terminal prompt is painful: we write a draft file
        # inside reports/, you edit it in your editor, then press Enter.
        draft_file = ct.REPORTS_DIR / "_draft_to_edit.md"
        ct.REPORTS_DIR.mkdir(exist_ok=True)
        draft_file.write_text(draft["content_markdown"])
        input(f"✏️  Edit {draft_file.relative_to(ct.ROOT)}, save it, then press Enter...")
        content = draft_file.read_text()
        draft_file.unlink()
        return {"action": "edit", "content": content}
    return {"action": "reject", "feedback": input("Why? (feedback for the agent): ")}


# ---------------------------------------------------------------------------
# 6. Run one turn of the conversation, handling interrupts until the graph ends.
# ---------------------------------------------------------------------------
def run_turn(user_text: str, thread_id: str, first_turn: bool,
             answer_fn=answer_in_terminal, approve_fn=approve_in_terminal, verbose: bool = True) -> dict:
    """Send one user message on a thread. Returns a trace (same shape as v0)."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    content = wrap_description(user_text) if first_turn else user_text
    # New turn: counters and report are reset; messages are APPENDED to the thread's history.
    graph_input = {"messages": [HumanMessage(content)], "questions_asked": 0,
                   "nudges": 0, "report_path": None}

    trace = {"tool_calls": [], "steps": 0, "nudges": 0, "final_answer": None,
             "report": None, "stopped_by": None, "latency_s": None}
    start = time.time()
    try:
        while True:
            interrupted = None
            # stream_mode="updates": we receive what each node returned, node by node.
            for update in graph.stream(graph_input, config, stream_mode="updates"):
                if "__interrupt__" in update:
                    interrupted = update["__interrupt__"][0].value
                    continue
                _log_update(update, trace, verbose)
            if interrupted is None:
                break  # the graph reached END
            # The graph is PAUSED (its state is in the checkpointer). Get the human's
            # input, then RESUME the graph with it.
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

    state = graph.get_state(config).values
    trace["final_answer"] = state["messages"][-1].content
    trace["latency_s"] = round(time.time() - start, 1)
    return trace


def _log_update(update: dict, trace: dict, verbose: bool) -> None:
    """Print Thought / Action / Observation, and fill the trace."""
    for node, values in update.items():
        if node == "agent":
            trace["steps"] += 1
            message = values["messages"][-1]
            if verbose:
                print(f"\n──────── Step {trace['steps']} (agent node) ────────")
                if message.content:
                    print(f"🧠 Thought: {message.content.strip()}")
                for call in message.tool_calls:
                    print(f"🔧 Action: {call['name']}({_short(json.dumps(call['args'], ensure_ascii=False), 200)})")
            trace["tool_calls"] += [{"name": c["name"], "arguments": c["args"], "result": None}
                                    for c in message.tool_calls]
        elif node == "tools":
            results = [json.loads(m.content) for m in values["messages"]]
            # attach each result to its call (the last len(results) calls of the trace)
            for call, result in zip(trace["tool_calls"][-len(results):], results):
                call["result"] = result
                if call["name"] == "save_report" and result.get("status") == "saved":
                    # read the saved file: if the human edited the draft, that is the real content
                    trace["report"] = {**call["arguments"], "path": result["path"],
                                       "content_markdown": (ct.ROOT / result["path"]).read_text()}
                if verbose:
                    icon = "❌" if "error" in result else "👀"
                    print(f"{icon} Observation: {_short(json.dumps(result, ensure_ascii=False), 300)}")
        elif node == "nudge":
            trace["nudges"] += 1
            if verbose:
                print("💬 Nudge: the report is not saved yet, reminding the agent.")


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


if __name__ == "__main__":
    if "--mermaid" in sys.argv:
        print(graph.get_graph().draw_mermaid())
        sys.exit()

    thread_id = str(uuid.uuid4())  # one thread = one conversation, remembered by the checkpointer
    user_text = " ".join(sys.argv[1:]) or input("Describe the AI system to assess:\n> ")
    first_turn = True
    while user_text.strip():
        trace = run_turn(user_text, thread_id, first_turn)
        print(f"\n🏁 {trace['final_answer'] or '(no final answer)'}")
        print(f"   steps={trace['steps']}  stopped_by={trace['stopped_by']}  latency={trace['latency_s']}s")
        if trace["report"]:
            print(f"   report saved to {trace['report']['path']}")
        first_turn = False
        user_text = input("\n💬 Follow-up on the same thread (empty = quit): ")
