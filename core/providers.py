"""
core/providers.py

Ready-made think functions for common LLM providers.
Each think factory accepts list[Message] and converts to its native SDK format
internally. All return LLMResponse — the only contract the loop cares about.

Usage:
    from core.providers import openai_think, anthropic_think, gemini_think

    # OpenAI
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    think = openai_think(client, model="gpt-4.1", tools=TOOLS)

    # Anthropic
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    think = anthropic_think(client, model="claude-opus-4-7", tools=TOOLS)

    # Gemini
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    think = gemini_think(client, model="gemini-2.5-flash", tools=TOOLS)
"""

import json
from core.types import LLMResponse, ToolCall, Message

OPENAI_BASE_URL = "https://api.openai.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


# ── OpenAI ─────────────────────────────────────────────────────────────────────

def _to_openai_messages(messages: list) -> list[dict]:
    result = []
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            result.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })
        elif msg.role == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            })
        else:
            result.append({"role": msg.role, "content": msg.content})
    return result


def openai_think(client, model: str, tools: list):
    async def think(messages: list) -> LLMResponse:
        resp = await client.chat.completions.create(
            model=model,
            messages=_to_openai_messages(messages),
            tools=tools,
        )
        msg = resp.choices[0].message
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in (msg.tool_calls or [])
        ]
        return LLMResponse(content=msg.content or "", tool_calls=tool_calls)
    return think


# ── Anthropic ──────────────────────────────────────────────────────────────────

def _to_anthropic_messages(messages: list) -> tuple[str, list[dict]]:
    system = ""
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role == "system":
            system = msg.content
            i += 1
            continue

        if msg.role == "user":
            result.append({"role": "user", "content": msg.content})
            i += 1
            continue

        if msg.role == "assistant":
            content = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            result.append({"role": "assistant", "content": content})
            i += 1
            continue

        if msg.role == "tool":
            # Consecutive tool results belong in one user message
            parts = []
            while i < len(messages) and messages[i].role == "tool":
                m = messages[i]
                parts.append({
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                })
                i += 1
            result.append({"role": "user", "content": parts})
            continue

        i += 1

    return system, result


def anthropic_think(client, model: str, tools: list):
    async def think(messages: list) -> LLMResponse:
        system, converted = _to_anthropic_messages(messages)
        resp = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=converted,
            tools=tools,
        )
        content = next((b.text for b in resp.content if b.type == "text"), "")
        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=b.input)
            for b in resp.content if b.type == "tool_use"
        ]
        return LLMResponse(content=content, tool_calls=tool_calls)
    return think


# ── Gemini ─────────────────────────────────────────────────────────────────────

def _to_gemini_messages(messages: list) -> tuple[str, list[dict]]:
    system = ""
    contents = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role == "system":
            system = msg.content
            i += 1
            continue

        if msg.role == "user":
            contents.append({"role": "user", "parts": [{"text": msg.content}]})
            i += 1
            continue

        if msg.role == "assistant":
            parts = []
            if msg.content:
                parts.append({"text": msg.content})
            for tc in msg.tool_calls:
                parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})
            contents.append({"role": "model", "parts": parts})
            i += 1
            continue

        if msg.role == "tool":
            # Consecutive tool results belong in one user turn
            parts = []
            while i < len(messages) and messages[i].role == "tool":
                m = messages[i]
                parts.append({
                    "functionResponse": {
                        "name": m.tool_name,
                        "response": {"result": m.content},
                    }
                })
                i += 1
            contents.append({"role": "user", "parts": parts})
            continue

        i += 1

    return system, contents


def gemini_think(client, model: str, tools: list):
    gemini_tools = [{"function_declarations": tools}] if tools else []

    async def think(messages: list) -> LLMResponse:
        system, contents = _to_gemini_messages(messages)

        config = {"tools": gemini_tools}
        if system:
            config["system_instruction"] = system

        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        content = ""
        tool_calls = []

        for idx, part in enumerate(candidate.content.parts):
            if getattr(part, "text", None):
                content += part.text
            fc = getattr(part, "function_call", None)
            if fc:
                tool_calls.append(ToolCall(
                    id=getattr(fc, "id", None) or f"call_{fc.name}_{idx}",
                    name=fc.name,
                    arguments=dict(fc.args),
                ))

        return LLMResponse(content=content, tool_calls=tool_calls)

    return think
