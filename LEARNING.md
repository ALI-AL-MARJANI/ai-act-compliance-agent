# LEARNING.md — study notes

My notes from building this project: a course on AI agents for someone who already knows NLP and RAG well. Every example comes from the code in this repo or from a real run.

## Contents
1. [Agent vs LLM vs RAG vs workflow](#1-agent-vs-llm-vs-rag-vs-workflow)
2. [Tool calling](#2-tool-calling)
3. [ReAct: reason → act → observe](#3-react-reason--act--observe)
4. [Stop conditions, infinite loops, MAX_STEPS](#4-stop-conditions-infinite-loops-max_steps)
5. [State, memory and checkpointers](#5-state-memory-and-checkpointers-in-langgraph)
6. [Nodes, edges, conditional edges](#6-nodes-edges-conditional-edges)
7. [Human-in-the-loop](#7-human-in-the-loop)
8. [MCP](#8-mcp)
9. [Multi-agent patterns (theory)](#9-multi-agent-patterns-theory-for-the-next-iteration)
10. [Evaluating an agent](#10-evaluating-an-agent)
11. [Agent safety](#11-agent-safety)
12. [The EU AI Act in 10 bullet points](#12-the-eu-ai-act-in-10-bullet-points)
13. [Failure modes observed in this project](#13-failure-modes-observed-in-this-project)
14. [20 interview questions + model answers](#14-20-interview-questions--model-answers)

---

## 1. Agent vs LLM vs RAG vs workflow

| | What it does | Who decides the next step? |
|---|---|---|
| **LLM** | text in → text out, one call | nobody: there is no next step |
| **RAG** | retrieve documents, then generate an answer with them | the developer (fixed: retrieve → generate) |
| **Workflow / pipeline** | several LLM calls and tools in a fixed order | the developer (the code) |
| **Agent** | an LLM in a loop that chooses tools and sees their results | **the LLM** |

The key difference is **who controls the flow**. In a workflow, the code says "first call A, then B". In an agent, the LLM reads the situation and decides: "I'll call `check_high_risk_areas('employment')` now", "I need to ask the user", "I'm done".

**Why this project is an agent:** the path depends on the case. For a chatbot, the agent goes prohibited → high-risk (no match) → Article 50 → limited risk. For a credit model, it stops at high-risk. For a vague description, it should ask a question first. A fixed pipeline would need many `if`s for all these paths. Here the LLM chooses them.

**The honest counterpoint:** the procedure (prohibited → high-risk → transparency → minimal) is quite fixed. A workflow with one LLM call per step would probably be **more reliable** with a 7B model. The agent version is more flexible (asking questions, retrying after an error, handling follow-ups), but less predictable. That trade-off is the key design question.

## 2. Tool calling

The LLM never runs code. Here is what really happens (see `v0_scratch/agent.py`):

1. We send the LLM the conversation **plus a JSON schema for each tool**: its name, a description and its typed parameters. In this repo the schema is generated from the pydantic model, `GetArticleArgs.model_json_schema()`.
2. The model was fine-tuned to answer, when useful, with a **structured tool call** instead of text:
   `{"name": "get_article", "arguments": {"regulation": "gdpr", "article": "22"}}`
3. **Our code** runs `get_article("gdpr", "22")`.
4. We append the result to the conversation as a message with `role: "tool"`, and call the LLM again.

So "tool calling" = the model produces structured output that our code turns into a real function call. The tool's **description** and the **error messages** are effectively prompts. Write them for the model: "Available articles: 5, 6, 9, 22, 35" helps it recover.

## 3. ReAct: reason → act → observe

ReAct (Yao et al., 2022) = interleave **Reasoning** (a thought), **Acting** (a tool call) and **Observing** (the tool result) until the task is done. In the terminal output:

```
🧠 Thought: The system matches use case 4(a) recruitment...      ← reason
🔧 Action: get_obligations({"risk_category": "high_risk"})       ← act
👀 Observation: {"obligations": [{"citation": "[AI Act Art. 9]" ← observe
```

The observation gives the model **new information** it did not have (the knowledge base, the user's answer, an error), and the next thought uses it. Note: qwen2.5 writes no thought before tool calls by default. I had to ask for "one short sentence explaining why" in the prompt.

## 4. Stop conditions, infinite loops, MAX_STEPS

An agent loop needs two kinds of stop:

- **Normal stop:** the LLM answers **without** a tool call → it's the final answer.
- **Safety stop:** `MAX_STEPS = 15` in v0, `recursion_limit` in v1. The LLM may never decide to stop: it can repeat the same call, or two tools can bounce errors back and forth. Without a limit, you burn time and money forever.

A third problem I did not expect: **stopping too early**. In the eval, 23 of 26 runs ended their turn *before* saving a report at least once. The fix is a **nudge**: if the model stops and the report is not saved, add a message "You have not saved the report yet", at most `MAX_NUDGES = 2` times. That's a third stop rule: "stop only when the task is really done, or give up after N reminders".

## 5. State, memory and checkpointers in LangGraph

- **State** = a typed dict that every node reads (`State` in `v1_langgraph/agent.py`: `messages`, `questions_asked`, `nudges`, `report_path`). A node returns only an **update**.
- **Reducer** = how an update is merged. `messages: Annotated[list, add_messages]` means "append these messages" instead of "replace the list". Without a reducer, a field is simply overwritten.
- **Checkpointer** = saves the state after **every node**, under a `thread_id`. This gives:
  - **memory**: call the graph again with the same `thread_id` and the whole conversation is still there (the "what if a human reviews the ranking?" follow-up);
  - **pause/resume**: `interrupt` works because the state is saved before pausing.
- `InMemorySaver` = RAM only, lost when the program stops. In production you'd use a Postgres or SQLite saver.

"Memory" here is **short-term** (one conversation). Long-term memory (across conversations, like user preferences) is a different thing: a store you read and write explicitly.

## 6. Nodes, edges, conditional edges

```mermaid
graph TD
    START --> agent
    agent -.->|tool calls| tools
    agent -.->|stopped too early| nudge
    agent -.->|done| END
    tools --> agent
    nudge --> agent
```

- **Node** = a plain Python function `state → update` (`agent_node`, `tools_node`, `nudge_node`).
- **Edge** = "after A, always go to B" (`tools → agent`).
- **Conditional edge** = a function that looks at the state and returns the next node's name (`route_after_agent`: tool calls → `tools`; no report yet → `nudge`; otherwise → `END`).

This is exactly the v0 `while` loop, but **explicit**: you can draw it (`--mermaid`), checkpoint it and pause it.

## 7. Human-in-the-loop

**Naive (v0):** `input("Approve? (y/n)")` inside the loop. It works in a terminal, but the program blocks, the state lives only in memory, and nobody can answer from a web page, from Slack, or tomorrow.

**With `interrupt` (v1):**
1. A node calls `interrupt({"type": "approval", ...})`.
2. LangGraph saves the state (checkpointer) and stops. Our code receives the payload.
3. Later (seconds or days), we call `graph.stream(Command(resume=answer), config)`.
4. ⚠️ The node **runs again from the start**, and this time `interrupt()` returns `answer`.

Consequence of point 4: code *before* an `interrupt` in the same node runs twice, so it must be safe to repeat. That's why `questions_asked` is only increased **after** the answer. Known limitation: if one LLM message contains `save_report` and then `ask_user`, the save would run again on resume.

**Two different kinds of human-in-the-loop:**

| | Clarification (`ask_user`) | Approval (`save_report`) |
|---|---|---|
| Purpose | get **missing facts** | gate an **irreversible action** |
| Who triggers it | the **LLM** decides to ask | the **code** forces it, every time |
| If the LLM forgets | it guesses (see §13) | impossible: it's not optional |

## 8. MCP

**In one sentence:** MCP (Model Context Protocol) is an open standard through which any AI application (the client) discovers and uses the tools, data and prompts offered by any server, so you write an integration once and plug it into many clients.

- **Client / server:** `v2_mcp/agent.py` (and Claude Desktop) are clients, `v2_mcp/server.py` is the server.
- **Transport:** *stdio* = the client starts the server as a subprocess and they exchange JSON-RPC messages over stdin/stdout. That's why the server must never `print()`, and why `ask_user` (which uses `input()`) can't live there. *Streamable HTTP* = the server runs remotely (e.g. a container on Azure), with authentication.
- **The three kinds of things a server offers:**

| | Example here | Who controls it |
|---|---|---|
| **Tool** | `get_article(regulation, article)` | the **model** decides to call it |
| **Resource** | `kb://gdpr/article/22` | the **application** reads it (by URI) |
| **Prompt** | `assess_ai_system(description)` | the **user** picks it |

- MCP also has **elicitation** (the server asks the user a question through the client). It could replace my local `ask_user`; I kept it simple for v1.
- **Why it's becoming a standard:** before MCP, each framework had its own tool format, so N clients × M tools = N×M integrations. With one protocol it's N + M. It's supported by Anthropic, OpenAI, Microsoft, Google and many IDEs.
- **Why it matters in a company:** one compliance tool server that the compliance team owns, with logs and access control in one place, used by many agents and apps. The agent builders don't need to know how the tools work.

## 9. Multi-agent patterns (theory, for the next iteration)

- **Router:** one LLM call classifies the request and sends it to *one* specialised agent (e.g. "AI Act question" → AI Act agent, "GDPR question" → GDPR agent).
- **Supervisor:** a supervisor agent calls worker agents *as tools*, reads their results, and decides what's next. The plan for this repo: an AI Act agent, a GDPR agent and a report writer, under a supervisor.
- **Hierarchical:** supervisors of supervisors (teams of teams), for large tasks.
- **Network / swarm:** agents hand off directly to each other, with no central boss.

**When it helps:** each agent gets a short, focused prompt and few tools, and small models get much better when the prompt is short. **What it costs:** more LLM calls (latency), harder debugging, and errors that compound across agents. Start with one agent. Split only when evaluation shows a clear problem (e.g. GDPR analysis gets forgotten, as in this project: GDPR flag recall = 0.47).

## 10. Evaluating an agent

Two levels:
- **Final answer:** is the risk category right? Are the cited articles right?
- **Trajectory:** *how* did it get there? Did it ask before concluding, how many steps did it take, did it hit errors and self-correct, did it need nudges?

Both matter: an agent can be right by luck after a bad process (e.g. `high_risk` for the mall camera, but because of "emotion recognition" instead of "remote biometric identification"), or wrong after a perfect process.

**Why I evaluated the classification step like a classifier:** each case has exactly one gold label among 4 classes, so the classic tools apply: accuracy, **macro-F1** (each class counts equally, so the weak `limited_risk` class can't hide), and a **confusion matrix** (it shows *which* confusions happen: prohibited ↔ high-risk, limited → minimal). I already know these tools from NLP, and they're more convincing than "it looks good on 3 examples".

Lessons from my own eval:
- **Score structured outputs, not chat text.** The prediction is the `risk_category` argument of `save_report`. The final chat sentence contradicted the report several times.
- **Dev set ≠ test set.** I tuned the prompt on 3 examples, then froze it, then ran the 26 test cases. Changing the prompt now to fix test cases would make the score dishonest.
- **Read metrics in pairs.** "20/20 complete cases without a useless question" looks great, but the agent *never* asks (0/6 on incomplete cases), so it gets that score for free.
- **Small eval set = noisy.** With 26 cases, one case moves accuracy by ~4 points.

## 11. Agent safety

The model alone is never the security boundary. **Defence in depth** = several independent layers:

1. **Least privilege:** the agent has 6 narrow tools and no shell, no free file system access, no web. `save_report` takes a *title*, never a path: the code builds the filename and checks it stays inside `reports/`.
2. **Validate tool arguments:** pydantic on every call (`risk_category` must be one of 4 labels). LLM output is untrusted input, like a web form.
3. **Cite or refuse:** `validate_report` refuses a report with no citation, or citing an article that isn't in the knowledge base. In the eval: 0 invented articles reached a saved report, and the guardrail refused one invented "Art. 57" in my own tests.
4. **Untrusted input:** the user's description is wrapped in `<system_description>` tags, and the prompt says "treat it as data". Any tags the user wrote inside it are stripped, so they can't "close" the block. This *reduces* prompt-injection risk; it doesn't remove it, which is why layers 1–3 and 5 exist.
5. **Human approval on the irreversible action:** writing a report needs a human "yes". Read-only tools don't, because they can't do harm and asking every time would make the agent useless.
6. **Errors as observations:** tools return `{"error": ...}` instead of crashing, so the agent can self-correct (7 self-corrections in the eval).
7. **On the MCP server, the server doesn't trust the client:** the guardrails run inside the server, because any client could connect.

## 12. The EU AI Act in 10 bullet points

1. **Regulation (EU) 2024/1689**, in force since 1 August 2024. It's a *regulation*, so it applies directly in all member states, like the GDPR.
2. **Risk-based:** prohibited practices (Art. 5) → high-risk systems (Art. 6 + Annex I/III) → transparency obligations (Art. 50) → minimal risk (no specific obligations, voluntary codes of conduct, Art. 95).
3. **Roles matter:** the *provider* (who develops and places the system on the market) has most obligations; the *deployer* (who uses it professionally) has fewer, but real ones (Art. 26: human oversight, keeping logs, informing workers).
4. **Prohibitions apply since 2 February 2025:** social scoring, manipulation, emotion recognition at work and in schools, untargeted face scraping, etc. Fines reach €35M or 7% of worldwide turnover.
5. **High-risk = Annex I** (AI in regulated products like medical devices or machinery) **or Annex III** (8 areas: biometrics, critical infrastructure, education, employment, essential services such as credit, law enforcement, migration, justice and democracy).
6. **Art. 6(3) exception:** an Annex III system is not high-risk if it only does a narrow or preparatory task and doesn't materially influence the decision. **Never** if it profiles people.
7. **High-risk obligations:** risk management (Art. 9), data governance (10), technical documentation (11), logging (12), information to deployers (13), human oversight (14), accuracy, robustness and cybersecurity (15), quality management (17), conformity assessment (43), registration (49).
8. **Timeline changes:** the *Digital Omnibus on AI* (Regulation (EU) 2026/1744) postponed Annex III high-risk obligations to **2 December 2027** and Annex I to **2 August 2028**; transparency (Art. 50) applies from 2 August 2026. Check the current timeline on EUR-Lex; see `knowledge/SOURCES.md`.
9. **General-purpose AI models** (like LLMs) have their own obligations since 2 August 2025: documentation, copyright policy, and more for "systemic risk" models.
10. **The AI Act does not replace the GDPR.** Most AI systems that touch people also need a legal basis (GDPR Art. 6), care with special categories (Art. 9) and with fully automated decisions (Art. 22), and often a DPIA (Art. 35), which can be combined with the AI Act's fundamental rights impact assessment (Art. 27).

## 13. Failure modes observed in this project

All real, from the runs during development and from `eval/results.json` (qwen2.5:7b).

| # | Failure | Real example | Mitigation |
|---|---|---|---|
| 1 | **Guessing instead of asking** | "An AI tool used by an HR department" → *"it likely involves analyzing and filtering applications"* → `high_risk`. 0/6 questions asked on incomplete cases. | Not solved. Next idea: an explicit "what facts are missing?" node before classifying. |
| 2 | **Stopping too early** | The model writes a text answer before saving the report (23/26 eval runs). | Nudge (max 2). |
| 3 | **Claiming an action without doing it** | Thought: *"Report saved successfully."*, but no `save_report` call. | The nudge checks the *real* state (`report_path`), not what the model says. |
| 4 | **Invented article** | Cited "AI Act Art. 57" (a real article about sandboxes, but not in the knowledge base and irrelevant). | `validate_report` refused it; the model removed it. |
| 5 | **Misusing a legal exception** | The CV-ranking tool classified `limited_risk` because "it does not materially influence decisions" (Art. 6(3)), while it auto-rejects 50% of candidates. | Moved the rule *into the tool output* ("the exception never applies if it ranks or profiles people"). Fixed on that example. |
| 6 | **"Human review" ≠ not high-risk** | Follow-up "what if a recruiter reviews the ranking?" → `limited_risk`, with no tool call. In fact it's still Annex III 4(a); human review only matters for GDPR Art. 22. | Not solved. Documented. |
| 7 | **Reading only part of a rule** | Chatbot → `minimal_risk` because "it doesn't generate synthetic content", ignoring Art. 50(1) (interaction with people). In the eval, `limited_risk` is the weakest class (1/6). | An explicit "ANY of these" list in the prompt fixed the dev example, not the eval cases. |
| 8 | **Batching arguments** | `get_article(article="5,6,9,22")`. My validator first kept "5" silently. | The validator now refuses ("ONE article per call") → the model retries. |
| 9 | **Final text contradicts the structured output** | Report saved as `high_risk`, final sentence "prohibited". | The eval reads the report's `risk_category`, never the chat text. |
| 10 | **Right label, wrong reason** | Mall face recognition → `high_risk` because of "emotion recognition (1(c))" instead of "remote biometric identification (1(a))". | Measured by the article metrics, not by accuracy. |
| 11 | **Prompt sensitivity** | Adding a bullet list about Art. 50 flipped the CV case from `high_risk` to `limited_risk` (temperature 0). | Freeze the prompt, measure on a test set. |

Big lesson: **the agent code worked; the model's reasoning was the weak link.** The disclaimer check, the invented-article check and the argument validation all fired in real runs. The path-traversal check and the `ask_user` limit never fired for real (the model never tried); they are covered by unit tests only.

## 14. 20 interview questions + model answers

**1. What's your experience with AI agents?**
I built a compliance assessment agent three times: a hand-written tool-calling loop (v0), the same agent in LangGraph with a checkpointer and `interrupt` for human-in-the-loop (v1), and a version that loads its tools from an MCP server (v2). Then I benchmarked it on 26 labelled cases. The baseline gets 46% accuracy with a local 7B model, and I can explain exactly where it fails.

**2. What is an agent, in one sentence?**
An LLM in a loop that decides which tool to call next, sees the result, and continues until the task is done. The model controls the flow, not the code.

**3. Why does your loop stop?**
Two conditions: the model answers without a tool call (normal end), or a step limit is reached (`MAX_STEPS = 15` in v0, `recursion_limit` in LangGraph). There's also a third rule: if the model stops before saving the report, I nudge it, at most twice.

**4. What happens if the model invents an article number?**
`save_report` extracts every citation with a regex and checks it against the knowledge base. An unknown article means the report is refused, and the error message goes back to the model, which removes it and retries. This happened for real with "Art. 57". In the eval, 0 invented articles reached a saved report.

**5. Why LangGraph instead of your own loop?**
My loop was fine for the logic. LangGraph adds three things that are painful to build myself: persistence (checkpointer + `thread_id`), pause and resume (`interrupt`), and an explicit, drawable control flow. The cost is a fast-changing dependency and more abstraction. My eval showed identical results for v0 and v1: LangGraph improves the engineering, not the reasoning.

**6. What is MCP in one sentence?**
An open protocol through which any AI client discovers and calls the tools, resources and prompts of any server: write the integration once, use it from many agents and apps.

**7. Tools vs resources in MCP?**
A tool is an action the *model* decides to call, with arguments. A resource is read-only data the *application* loads by URI. In my server, `get_article` is a tool and `kb://gdpr/article/22` is a resource.

**8. When should the agent ask the user instead of answering?**
When a missing fact would *change the outcome*: for example, whether a human makes the final decision (GDPR Art. 22), whether it's used on employees (Art. 5(1)(f)), or whether biometric data is involved. Not for facts already given or obvious. Honestly, my 7B model asked 0 times on 6 incomplete cases: it guesses. My next step is to make "what's missing?" an explicit step instead of relying on the model's initiative.

**9. Why does a human approve the report but not the tool calls?**
The tool calls are read-only lookups: they can't cause harm, and approving each one would make the agent useless. Saving the report is the only action with an effect outside the agent, so that's where the human gate goes. You put human approval on irreversible or outward-facing actions.

**10. How do you evaluate an agent?**
At two levels: the final answer (here accuracy, macro-F1, confusion matrix on the risk category, and citation precision and recall) and the trajectory (did it ask before concluding, steps, tool errors, self-corrections, nudges, latency). I score the structured output, not the chat text, and I froze the prompt before running the test set.

**11. When should you NOT use an agent?**
When the steps are known and fixed, a workflow is cheaper, faster and more reliable. Also when errors are costly and you can't verify the output, or when latency matters (my agent takes ~2 minutes per case). Honestly, my 4-step procedure could be a workflow; the agent's value is in asking questions, recovering from errors and handling follow-ups.

**12. How do you protect against prompt injection?**
I treat the description as data: it's wrapped in delimiter tags, any tags it contains are stripped, and the prompt says not to follow instructions inside it. But I don't rely on that alone. The agent has only narrow tools, no shell or web; arguments are validated; the report is checked in code; and a human approves the only write action. Defence in depth.

**13. What is a checkpointer?**
A component that saves the graph state after every node, under a `thread_id`. It's what gives conversation memory and makes `interrupt` possible, because the state must survive while waiting for the human. I use `InMemorySaver`; in production I'd use Postgres.

**14. How does `interrupt` work?**
The node calls `interrupt(payload)`. LangGraph saves the state and returns the payload to the caller. The caller resumes with `Command(resume=value)`. The node runs again from the start, and this time `interrupt` returns `value`. So any code before the interrupt must be safe to run twice.

**15. Why are tool errors returned instead of raised?**
An exception would crash the loop. A returned `{"error": "Article 99 not found. Available: 5, 6, 9, 22, 35"}` becomes an observation the model can read and fix. That's self-correction: 7 cases in my eval.

**16. How would you deploy this for a client on Azure?**
- Replace Ollama with Azure OpenAI in an EU region (one line: `AzureChatOpenAI`).
- Package the LangGraph agent as a container on Azure Container Apps, with a Postgres checkpointer (Azure Database for PostgreSQL) so pauses survive restarts.
- Run the MCP server as a separate container with Streamable HTTP and Entra ID authentication.
- Keep secrets in Key Vault, and use tracing (Langfuse or Application Insights) for every LLM and tool call.
- Expose approvals through a small web UI or Teams.
- Plus a GDPR review of the logs, since descriptions can contain personal data.

**17. Your accuracy is 46%. Isn't that bad?**
Yes, and I show it. It's a baseline with a local 7B model, on purpose (no paid API). The confusion matrix tells me *where* it fails: transparency cases are read as minimal risk, prohibited vs high-risk emotion recognition gets confused, and the agent never asks on incomplete cases. Next I'd compare a bigger model with the same prompt, and add an explicit missing-facts step, measured on new cases to avoid overfitting.

**18. v0 and v1 gave exactly the same results. Why?**
Same model, same prompt, same tool schemas, temperature 0. The framework only changes the loop around the model. The only difference is latency: v1 was ~40% slower (175 s vs 125 s), because of LangChain's overhead and checkpointing.

**19. Why a small JSON knowledge base and not RAG?**
The knowledge needed is small (24 articles, 8 Annex III areas) and well structured by article number. Lookup by key is exact and testable, and it allows the "cite or refuse" check. RAG adds retrieval errors for no benefit at this size. I've built RAG systems before; this project is about agent reasoning.

**20. What would you do with more time?**
1. Compare model sizes (7B vs 14B) with the same prompt.
2. Add a missing-facts step and score it on new cases.
3. Split into a supervisor with AI Act, GDPR and report-writer agents, since GDPR recall is 0.47.
4. Add tracing with Langfuse.
5. Deploy on Azure.
6. Get a lawyer-reviewed eval set.
7. Compare a fine-tuned small classifier against the agent for the classification step alone. My guess is the classifier wins on accuracy, and the agent wins on explanations and questions.
