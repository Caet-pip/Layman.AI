"""
core/loop.py

The agent loop. Framework-agnostic, model-agnostic.
Bring your own LLM client, tools, and state — this runs the loop.

Usage:
    from core.loop import run

    result = await run(
        task="find me something",
        state=state,
        think=my_think,
        execute=my_execute,
        judge=my_judge,         # optional
        max_steps=100,
    )
"""

from core.heal import heal
from core.types import Message

MAX_STEPS = 100


async def run(
    task: str,
    state,
    think,
    execute,
    judge=None,
    max_steps: int = MAX_STEPS,
) -> str:
    """
    Core agent loop.

    Args:
        task      — the user's request
        state     — AgentState instance (see state/base.py)
        think     — async fn(messages: list[Message]) -> LLMResponse
        execute   — async fn(tool_call: ToolCall, task: str) -> dict | None
                    returns {"content": "..."} (tool_call_id is inferred from ToolCall)
        judge     — optional async fn(task, answer) -> (bool, str)
        max_steps — hard cap on loop iterations

    Returns:
        The agent's final answer as a string.
    """
    state.messages.append(Message(role="user", content=task))

    for step in range(max_steps):
        heal(state.messages)

        print(f"[loop] step {step + 1} — thinking...", flush=True)
        response = await think(state.messages)

        # ── No tool calls: agent has an answer ──
        if not response.tool_calls:
            print(f"[loop] step {step + 1} — no tool calls, returning answer", flush=True)
            state.messages.append(Message(role="assistant", content=response.content))

            if judge:
                print(f"[loop] running judge...", flush=True)
                sufficient, feedback = await judge(task, response.content)
                if not sufficient:
                    print(f"[loop] judge rejected: {feedback}", flush=True)
                    state.messages.append(Message(
                        role="user",
                        content=f"[Judge feedback] {feedback} Please continue.",
                    ))
                    continue

            return response.content

        # ── Tool calls: execute all, commit atomically ──
        # Never append assistant message without its tool results — broken pairs confuse the LLM
        tool_names = [tc.name for tc in response.tool_calls]
        print(f"[loop] step {step + 1} — tool calls: {tool_names}", flush=True)

        results = []
        for tc in response.tool_calls:
            print(f"[loop] executing: {tc.name} | args: {tc.arguments}", flush=True)
            result = await execute(tc, task)
            if result is not None:
                results.append((tc, result))

        state.messages.append(Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls,
        ))
        for tc, r in results:
            state.messages.append(Message(
                role="tool",
                content=r["content"],
                tool_call_id=tc.id,
                tool_name=tc.name,
            ))

    return "Reached max steps without completing the task."
