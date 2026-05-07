"""
core/judge.py

Optional quality gate — runs after the agent proposes an answer.
If the answer isn't sufficient, the loop continues with feedback.

The judge is a separate LLM call with a strict prompt. It's intentionally
decoupled from the main loop so you can swap it out or skip it entirely.

Usage:
    from laymanai.core.judge import make_judge

    judge = make_judge(llm_client, model="gpt-4.1", max_rounds=3)

    result = await run(
        task=task,
        state=state,
        think=think,
        execute=execute,
        judge=judge,
    )
"""

import json


def make_judge(client, model: str, max_rounds: int = 3, extra_criteria: str = ""):
    """
    Returns a judge function compatible with core/loop.py.

    Args:
        client         — LLM client (OpenAI-compatible)
        model          — model name to use for judging
        max_rounds     — maximum number of judge retries before accepting
        extra_criteria — optional task-specific criteria appended to the prompt

    Returns:
        async fn(task, answer) -> (sufficient: bool, feedback: str)
    """
    rounds = {"count": 0}

    async def judge(task: str, answer: str) -> tuple[bool, str]:
        if rounds["count"] >= max_rounds:
            return True, ""

        rounds["count"] += 1

        extra = f"\nAdditional criteria:\n{extra_criteria}" if extra_criteria else ""
        prompt = f"""You are a strict judge evaluating an AI agent's answer.

TASK: {task}
AGENT'S ANSWER: {answer}{extra}

Respond in JSON: {{"sufficient": true/false, "feedback": "one sentence"}}
Be strict. Vague or generic answers without specific details are NOT sufficient."""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get("sufficient", True), result.get("feedback", "")
        except Exception:
            return True, ""

    return judge
