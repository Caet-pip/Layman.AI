"""
core/judge.py

Optional quality gate — runs after the agent proposes an answer.
If the answer isn't sufficient, the loop continues with feedback.
"""

import json
from core.types import Message


DEFAULT_JUDGE_PROMPT = """You are a strict judge evaluating an AI agent's answer.

TASK: {task}
AGENT'S ANSWER: {answer}{extra}

Respond in JSON: {{"sufficient": true/false, "feedback": "one sentence"}}
Be strict. Vague or generic answers without specific details are NOT sufficient."""


def make_judge(think, max_rounds: int = 3, extra_criteria: str = "", prompt: str = DEFAULT_JUDGE_PROMPT):
    """
    Returns a judge function compatible with core/loop.py.

    Args:
        think          — async fn(messages) -> LLMResponse, same think used by the agent
                         passed in with no tools so the judge never calls any
        max_rounds     — maximum rejections before accepting
        extra_criteria — optional criteria appended to the prompt
        prompt         — override the full judge prompt template

    Returns:
        async fn(task, answer) -> (sufficient: bool, feedback: str)
    """
    count = 0

    async def judge(task: str, answer: str) -> tuple[bool, str]:
        nonlocal count
        if count >= max_rounds:
            return True, ""

        count += 1

        extra = f"\nAdditional criteria:\n{extra_criteria}" if extra_criteria else ""
        filled = prompt.format(task=task, answer=answer, extra=extra)

        response = await think([Message(role="user", content=filled)])
        try:
            result = json.loads(response.content)
            return result.get("sufficient", True), result.get("feedback", "")
        except Exception:
            return True, ""

    return judge
