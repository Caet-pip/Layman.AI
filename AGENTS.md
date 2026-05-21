# AGENTS.md — Guide for AI Assistants

This file is for AI coding assistants (Claude, Copilot, Gemini, etc.) working in this repo.

## What this project is

Layman.AI is an educational library of core agentic primitives built from scratch with no frameworks. The goal is that anyone learning how AI agents work can read every line and understand exactly what's happening. Simplicity and clarity are the top priority — always over cleverness or completeness.

## Philosophy — read this first

- **No frameworks.** No LangChain, LlamaIndex, AutoGen, or similar. Everything is plain Python.
- **No unnecessary abstractions.** If three lines of code are clear, do not wrap them in a class.
- **Educational over optimal.** Code should be easy to read and reason about, not maximally efficient.
- **Each file should be self-contained enough to understand on its own.** A learner should be able to open any file and follow it without needing to trace through five others.

## Project structure

```
core/types.py       — shared data types: Message, ToolCall, LLMResponse.
                      the neutral format the entire harness speaks.

core/loop.py        — the agent loop (think, execute, repeat). the heart of everything.

core/heal.py        — strips broken tool call pairs from the message list before LLM calls.

core/judge.py       — optional quality gate. uses the same think function as the agent.

core/agent.py       — the Agent class. wires client, model, provider, tools, and judge
                      into one object. this is what users instantiate.

core/providers.py   — think() factories for OpenAI, Anthropic, Gemini, Ollama.
                      each one has its own private message converter and speaks
                      its native SDK directly. no cross-provider dependencies.

core/tools.py       — define_tool() and bundle(tools, provider=...) for converting
                      tool schemas to each provider's format.

state/base.py       — AgentState dataclass. holds list[Message]. extend per project.

examples/           — complete working agents. each example is self-contained and
                      shows exactly how to wire Agent + tools + execute together.

architecture.html   — interactive browser visualisation of every file, what it defines,
                      and how data flows between them. open directly in a browser.
                      drag cards, zoom, click symbols to trace dependencies.
```

## Rules for contributing new code

- New primitives go in `core/` if they are part of the agent loop machinery.
- New examples go in `examples/` and must be fully working end to end.
- Every file must have a docstring at the top explaining what it does, why it exists, and how to use it.
- Do not add `__all__`, complex `__init__.py` exports, or metaclasses. Keep `__init__.py` files empty.
- Do not add logging frameworks, config systems, or CLI argument parsers beyond what is already in the examples.
- Do not add type stubs, abstract base classes, or protocol classes unless there is a clear reason.
- New providers go in `core/providers.py` — add a `_to_X_messages()` converter and a `X_think()` factory. Add the provider to `_THINK_FACTORIES` and `_TOOL_FORMATS` in `core/agent.py`.

## Execute return format

`execute(tool_call, task)` must return `{"content": "..."}` or `None`. The loop reads
`tool_call_id` from the `ToolCall` object directly — do not include it in the return dict.

## Message format

All conversation history is stored as `list[Message]` in `AgentState`. `Message` is defined in `core/types.py` and is the neutral format the loop works with. Each `think()` factory in `providers.py` converts this format to its native SDK format privately — the loop never sees provider-specific shapes.

## What is coming soon

These are planned additions — do not implement them unless asked:

- Memory — persist information across agent runs
- Context management — token budget tracking and message summarization for long-running agents
- Streaming support
- Tool error handling and retry patterns
- Multi-agent patterns (orchestrator + subagents)
- More example agents

## What to avoid

- Do not refactor working code for style unless asked.
- Do not add error handling for scenarios that cannot happen in normal use.
- Do not add comments explaining what code does — only add a comment if the WHY is non-obvious.
- Do not rename files or reorganize the directory structure without asking.
- Do not add tests unless asked — this is an educational library, not a production system.
- Do not convert between provider formats inside the loop — that belongs in providers.py.
