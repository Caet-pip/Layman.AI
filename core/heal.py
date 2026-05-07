"""
core/heal.py

Heals the message list before every LLM call.

The problem: if the agent crashes mid-step — timeout, exception, keyboard
interrupt — the message list can be left in a broken state:
    [assistant: calls tool_A]   ← no matching tool result
or
    [assistant: calls tool_A, tool_B]
    [tool: result for tool_A]   ← tool_B result missing

Sending a broken message list to the LLM causes unpredictable behaviour.
heal() strips any trailing incomplete pairs so the list is always valid.

Usage:
    from laymanai.core.heal import heal
    heal(state.messages)        # mutates in place, call before every think()
"""


def heal(messages: list[dict]) -> None:
    """
    Remove trailing incomplete tool call pairs from the message list.
    Mutates in place. Safe to call even when messages is clean.
    """
    while messages:
        last = messages[-1]

        # Trailing assistant message with tool calls but no results yet
        if last["role"] == "assistant" and last.get("tool_calls"):
            messages.pop()
            continue

        # Trailing tool result(s) — walk back and check the pair is complete
        if last["role"] == "tool":
            i = len(messages) - 1
            tool_ids_present = set()
            while i >= 0 and messages[i]["role"] == "tool":
                tool_ids_present.add(messages[i].get("tool_call_id"))
                i -= 1

            if i >= 0 and messages[i]["role"] == "assistant" and messages[i].get("tool_calls"):
                expected = {tc["id"] for tc in messages[i]["tool_calls"]}
                if tool_ids_present == expected:
                    break  # pair is complete, stop
                # partial results — strip everything back to before the assistant message
                while len(messages) > i:
                    messages.pop()
            else:
                break

        else:
            break
