"""
tools/ask_human.py

Universal tool — works in any agent, any environment.
Pauses the agent and routes a question to a human.

In CLI mode:    blocks on input()
In server mode: pass a callback (e.g. WebSocket ask_human handler)

Usage:
    from laymanai.tools import ask_human

    # Schema to pass to the LLM in your tools list
    TOOL_SCHEMA = ask_human.schema

    # In your execute() function:
    if name == "ask_human":
        return await ask_human.execute(tool_call, callback=my_callback)
"""


# Schema to register with the LLM
schema = {
    "type": "function",
    "function": {
        "name": "ask_human",
        "description": (
            "Ask the human user for input, clarification, or a decision. "
            "Use this when you are genuinely blocked and cannot proceed without human input. "
            "Do NOT use this to confirm routine steps or ask if you should continue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the human.",
                }
            },
            "required": ["question"],
        },
    },
}


async def execute(tool_call, callback=None) -> dict:
    """
    Execute an ask_human tool call.

    Args:
        tool_call — the tool call object from the LLM
        callback  — optional async fn(question: str) -> str
                    if None, falls back to blocking input()

    Returns:
        Tool result dict ready to append to messages.
    """
    import json
    args = json.loads(tool_call.function.arguments)
    question = args.get("question", "")

    print(f"\n[ask_human] {question}")

    if callback:
        answer = await callback(question)
    else:
        answer = input("Your answer: ").strip()

    print(f"[human] {answer}")

    return {
        "tool_call_id": tool_call.id,
        "content": answer,
    }
