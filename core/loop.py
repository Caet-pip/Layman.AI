"""
core/loop.py

The agent loop. Framework-agnostic, model-agnostic.
Bring your own LLM client, tools, and state — this runs the loop.

Usage:
    from laymanai.core.loop import run

    result = await run(
        task="find me something",
        state=state,
        think=my_llm_call,
        execute=my_tool_executor,
        judge=my_judge,         # optional
        max_steps=100,
    )
"""


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
        think     — async fn(messages: list[dict]) -> message
                    calls the LLM, returns object with .content and .tool_calls
        execute   — async fn(tool_call, task) -> dict | None
                    executes one tool call, returns result dict or None
        judge     — optional async fn(task, answer) -> (bool, str)
                    returns (sufficient, feedback)
        max_steps — hard cap on loop iterations

    Returns:
        The agent's final answer as a string.
    """
    state.messages.append({"role": "user", "content": task})

    for step in range(max_steps):
        # Heal before every LLM call — strips broken tool call pairs
        heal(state.messages)

        print(f"[loop] step {step + 1} — thinking...", flush=True)
        message = await think(state.messages)

        # ── No tool calls: agent has an answer ──
        if not message.tool_calls:
            print(f"[loop] step {step + 1} — no tool calls, returning answer", flush=True)
            state.messages.append({
                "role": "assistant",
                "content": message.content,
            })

            if judge:
                print(f"[loop] running judge...", flush=True)
                sufficient, feedback = await judge(task, message.content)
                if not sufficient:
                    print(f"[loop] judge rejected: {feedback}", flush=True)
                    state.messages.append({
                        "role": "user",
                        "content": f"[Judge feedback] {feedback} Please continue.",
                    })
                    continue

            return message.content

        # ── Tool calls: execute all, commit atomically ──
        # Never append assistant message without its tool results — broken pairs confuse the LLM
        tool_names = [tc.function.name for tc in message.tool_calls]
        print(f"[loop] step {step + 1} — tool calls: {tool_names}", flush=True)
        assistant_msg = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        }

        results = []
        for tc in message.tool_calls:
            print(f"[loop] executing: {tc.function.name}", flush=True)
            result = await execute(tc, task)
            if result is not None:
                results.append(result)

        state.messages.append(assistant_msg)
        for r in results:
            state.messages.append({
                "role": "tool",
                "tool_call_id": r["tool_call_id"],
                "content": r["content"],
            })

    return "Reached max steps without completing the task."


def heal(messages: list[dict]) -> None:
    """Strips trailing incomplete tool call pairs from the message list."""
    while messages:
        last = messages[-1]
        if last["role"] == "assistant" and last.get("tool_calls"):
            messages.pop()
        elif last["role"] == "tool":
            i = len(messages) - 1
            tool_ids = set()
            while i >= 0 and messages[i]["role"] == "tool":
                tool_ids.add(messages[i].get("tool_call_id"))
                i -= 1
            if i >= 0 and messages[i]["role"] == "assistant" and messages[i].get("tool_calls"):
                expected = {tc["id"] for tc in messages[i]["tool_calls"]}
                if tool_ids == expected:
                    break
                while len(messages) > i:
                    messages.pop()
            else:
                break
        else:
            break
