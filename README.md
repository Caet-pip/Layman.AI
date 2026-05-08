# Layman.AI Agents

Core agentic concepts, built from scratch. No frameworks, no magic.

If you've ever wondered what actually happens inside an AI agent — this is it.

---

## Start here

Run the example agent first. It browses a folder of files and answers questions about them:

```bash
pip install openai
export OPENAI_API_KEY=your_key
python examples/local_file_agent.py --dir ./some_folder
```

Or with a local model via [Ollama](https://ollama.com):

```bash
python examples/local_file_agent.py --dir ./some_folder --backend ollama --model llama3.2
```

Then read `examples/local_file_agent.py`. It's ~150 lines and shows exactly how to wire everything together.

---

## How an agent actually works

An agent is just a loop. The pattern is called **ReAct** (Reason + Act) — the LLM reasons about what to do, acts by calling a tool, observes the result, then reasons again. That cycle repeats until it has an answer.

```
you give it a task
  ↓
agent asks the LLM: "what should I do?"   ← Reason
  ↓
LLM says: "call this tool"
  ↓
agent runs the tool, gets a result         ← Act
  ↓
agent tells the LLM what the tool returned ← Observe
  ↓
LLM says: "call another tool" or "here's my answer"
  ↓
repeat until done
```

In code, that's `core/loop.py`:

```python
for step in range(max_steps):
    message = await think(messages)       # ask the LLM what to do next

    if not message.tool_calls:
        return message.content            # LLM has an answer, we're done

    # LLM wants to call tools — run them all, then tell the LLM what happened
    results = [await execute(tc) for tc in message.tool_calls]
    messages.append(assistant_message)
    messages.extend(tool_results)
    # loop again
```

That's literally it. Everything else in this library is optional machinery around that loop.

---

## The message list

The LLM doesn't have memory. Everything it knows comes from the messages you send it each turn. The message list grows as the agent works:

```
[system prompt]           ← who the agent is, what tools it has
[user: "find me X"]       ← the task
[assistant: call tool_A]  ← LLM decided to use a tool
[tool: result of tool_A]  ← what the tool returned
[assistant: call tool_B]  ← LLM decided to use another tool
[tool: result of tool_B]
[assistant: "Here's what I found..."]  ← final answer
```

Every time around the loop, you send the full list to the LLM. It reads all of it and decides what to do next.

---

## What each file does

```
core/loop.py        — the loop above, as a reusable function

core/heal.py        — if the agent crashes mid-step, the message list can be left
                      broken (e.g. an assistant message that called a tool, but no
                      tool result following it). heal() strips those broken pairs
                      before the next LLM call so the LLM isn't confused.
                      NOTE: this is specific to the OpenAI message format — the
                      role/tool_call_id structure is an OpenAI API convention.
                      Other APIs (Anthropic, Gemini) use different formats and
                      would need their own heal() implementation.

core/judge.py       — after the agent gives an answer, a second LLM call checks
                      if it's actually good. if not, the loop continues.

state/base.py       — holds the message list and current task. extend this
                      dataclass to add your own fields (e.g. visited URLs).

tools/ask_human.py  — a tool the LLM can call when it needs input from you.
                      Two modes:
                        CLI mode: the agent prints the question and your terminal
                          freezes waiting for you to type an answer. Simple, works
                          anywhere.
                        Server mode: instead of blocking the terminal, it calls a
                          callback function you provide — e.g. send the question over
                          a WebSocket to a browser UI, wait for the user to type there,
                          and return the answer. This is how you embed ask_human into
                          a web app without freezing a server thread.

context/              — coming soon: token budget tracking, pinned prefix strategy,
                        and message summarization for long-running agents.
```

---

## Building your own agent

Three things to define, then call `run()`:

**1. Tools** — what can the agent do? Define schemas (what the LLM sees) and executors (what actually runs):

```python
MY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "Does something useful.",
        "parameters": { ... }
    }
}

async def execute(tool_call, task):
    if tool_call.function.name == "my_tool":
        # do the thing
        return {"tool_call_id": tool_call.id, "content": result}
```

**2. Think** — wrap your LLM call:

```python
def make_think(client, model, tools):
    async def think(messages):
        response = client.chat.completions.create(model=model, messages=messages, tools=tools)
        return response.choices[0].message
    return think
```

**3. Run**:

```python
from core.loop import run
from core.judge import make_judge
from state.base import AgentState

state = AgentState(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
think = make_think(client, model, TOOLS)
judge = make_judge(client, model, max_rounds=2)

result = await run(task="your task here", state=state, think=think, execute=execute, judge=judge)
```

---

## Coming soon

This library is intentionally primitive right now. The concepts below are being added:

- **Context management** — token budget tracking, pinned prefix strategy, and message summarization so agents can run long tasks without hitting context limits
- **Streaming** — handling token-by-token output for real-time UIs
- **Structured output** — reliably getting JSON back from the LLM
- **Tool error handling** — what happens when a tool fails, and how the agent recovers
- **More examples** — different agents showing the loop is truly reusable across use cases
- **Multi-agent** — orchestrator and subagent patterns

Feedback and contributions welcome. If something is confusing or missing, open an issue.

---

## Why no frameworks?

Frameworks are great once you know what's happening underneath. This library is for understanding what's underneath — so that when you do use a framework, you know exactly what it's doing for you.
