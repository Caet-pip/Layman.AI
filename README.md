# Layman.AI Agents

Core agentic concepts, built from scratch. No frameworks, no magic.

If you've ever wondered what actually happens inside an AI agent — this is it.

---

## Architecture

Open `architecture.html` in any browser for an interactive dependency map of the entire codebase — every file, what it defines, what it imports, and how data flows between them.

- Click a category group inside a card to expand its imports
- Click an individual symbol to focus it — the source and destination cards fly to the centre of the screen with the connecting arrow highlighted
- Drag any card by its header to reposition it
- `+` / `−` or `Ctrl+scroll` to zoom

---

## Start here

Install dependencies and run the example agent. It browses a folder of files and answers questions about them:

```bash
pip install -r requirements.txt

# OpenAI
export OPENAI_API_KEY=your_key
python examples/local_file_agent.py --dir ./some_folder

# Gemini
export GEMINI_API_KEY=your_key
python examples/local_file_agent.py --dir ./some_folder --backend gemini

# Ollama (local model)
python examples/local_file_agent.py --dir ./some_folder --backend ollama --model gemma4:4b
```

Then read `examples/local_file_agent.py`. It's ~150 lines and shows exactly how to wire everything together.

---

## How an agent actually works

An agent is just a loop. The model reasons about what to do, acts by calling a tool, observes the result, then reasons again. That cycle repeats until it has an answer — this pattern is called **ReAct** (Reason + Act).

```
you give it a task
  ↓
agent asks the LLM: "what should I do?"   ← Reason
  ↓
LLM says: "call this tool"
  ↓
agent runs the tool, gets a result         ← Act
  ↓
agent tells the LLM what happened          ← Observe
  ↓
LLM says: "call another tool" or "here's my answer"
  ↓
repeat until done
```

---

## Building your own agent

Three things to define, then create an `Agent`:

**1. A client** — your LLM provider:

```python
from openai import AsyncOpenAI
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

**2. Tools** — what the agent can do:

```python
from core.tools import define_tool, bundle

MY_TOOL = define_tool(
    name="my_tool",
    description="Does something useful.",
    properties={"input": {"type": "string", "description": "The input."}},
    required=["input"],
)
```

**3. An executor** — what actually runs when a tool is called:

```python
async def execute(tool_call, task):
    if tool_call.name == "my_tool":
        result = do_something(tool_call.arguments["input"])
        return {"tool_call_id": tool_call.id, "content": result}
```

**Then wire it up:**

```python
from core.agent import Agent

agent = Agent(
    client=client,
    model="gpt-5.5-instant",
    provider="openai",
    tools=[MY_TOOL],
    system_prompt="You are a helpful assistant.",
    judge="Answer must be specific and cite sources.",  # optional quality gate
)

result = await agent.run(task="your task here", execute=execute)
```

---

## What each file does

```
core/types.py       — shared data types: Message, ToolCall, LLMResponse.
                      The neutral format everything in the harness speaks.

core/loop.py        — the agent loop. calls think, executes tools, builds
                      the conversation history. framework and model agnostic.

core/heal.py        — strips incomplete tool-call pairs from the message list
                      before each LLM call so the model is never confused by
                      a half-finished turn.

core/agent.py       — the Agent class. wires together client, model, provider,
                      tools, and judge into one object. this is what you
                      instantiate in your own project.

core/providers.py   — think() factories for each provider (OpenAI, Anthropic,
                      Gemini, Ollama). each one translates the neutral Message
                      format into the provider's native SDK format and back.

core/tools.py       — define tools once, convert to any provider's schema with
                      bundle(tools, provider="openai"|"anthropic"|"gemini").

core/judge.py       — after the agent gives an answer, a second LLM call checks
                      if it's actually good. if not, the loop continues with
                      feedback. uses the same provider as the agent.

state/base.py       — holds the conversation history as a list of Message
                      objects. extend this dataclass to add your own fields.

architecture.html   — interactive browser visualisation of every file's
                      dependencies and data flow. open directly in a browser.
```

---

## Supported providers

| Provider | Backend flag | Client |
|----------|-------------|--------|
| OpenAI | `--backend openai` | `AsyncOpenAI` |
| Gemini | `--backend gemini` | `google.genai.Client` |
| Ollama | `--backend ollama` | `AsyncOpenAI` (compat) |
| Anthropic | — | `AsyncAnthropic` |

---

## Coming soon

- **Memory** — let agents remember things across runs
- **Context management** — token budget tracking and message summarization so agents can run long tasks without hitting context limits
- **Streaming** — token-by-token output for real-time UIs
- **Tool error handling** — what happens when a tool fails and how the agent recovers
- **Multi-agent** — orchestrator and subagent patterns
- **More examples** — different agents showing the loop is reusable across use cases

---

## Why no frameworks?

Frameworks are great once you know what's happening underneath. This library is for understanding what's underneath — so that when you do use a framework, you know exactly what it's doing for you.
