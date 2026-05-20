"""
core/agent.py

The Agent class. Wires together client, model, provider, tools, and judge
into a single object. Everything that used to live in main() lives here.

Usage:
    from openai import AsyncOpenAI
    from core.agent import Agent
    from core.tools import define_tool

    MY_TOOL = define_tool(name="my_tool", description="...", properties={}, required=[])

    agent = Agent(
        client=AsyncOpenAI(api_key="..."),
        model="gpt-4.1",
        provider="openai",
        tools=[MY_TOOL],
        system_prompt="You are a helpful assistant.",
        judge="Answer must include specific details.",
    )

    result = await agent.run(task="do something", execute=my_execute)
"""

from core.loop import run, MAX_STEPS
from core.judge import make_judge, DEFAULT_JUDGE_PROMPT
from core.providers import openai_think, anthropic_think, gemini_think
from core.tools import bundle
from core.types import Message
from state.base import AgentState


_THINK_FACTORIES = {
    "openai":    openai_think,
    "anthropic": anthropic_think,
    "gemini":    gemini_think,
    "ollama":    openai_think,
}

_TOOL_FORMATS = {
    "openai":    "openai",
    "anthropic": "anthropic",
    "gemini":    "gemini",
    "ollama":    "openai",
}


class Agent:
    def __init__(
        self,
        client,
        model: str,
        provider: str,
        tools: list,
        system_prompt: str = "",
        judge=False,
        max_judge_rounds: int = 3,
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
        max_steps: int = MAX_STEPS,
    ):
        """
        Args:
            client           — provider SDK client (AsyncOpenAI, AsyncAnthropic, genai.Client)
            model            — model name (e.g. "gpt-4.1", "gemini-2.5-flash")
            provider         — "openai" | "anthropic" | "gemini" | "ollama"
            tools            — list of define_tool() dicts
            system_prompt    — system prompt for the agent
            judge            — False to disable, True for default criteria,
                               or a string of extra criteria to append
            max_judge_rounds — max rejections before judge accepts
            judge_prompt     — override the full judge prompt template
            max_steps        — hard cap on loop iterations
        """
        if provider not in _THINK_FACTORIES:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_THINK_FACTORIES)}")

        self.client   = client
        self.model    = model
        self.provider = provider

        self._system_prompt = system_prompt
        self._max_steps     = max_steps
        self._tools         = bundle(tools, provider=_TOOL_FORMATS[provider])
        self._think         = _THINK_FACTORIES[provider](client, model, self._tools)
        self._judge         = None

        if judge is not False:
            extra = judge if isinstance(judge, str) else ""
            judge_think = _THINK_FACTORIES[provider](client, model, [])
            self._judge = make_judge(
                judge_think,
                max_rounds=max_judge_rounds,
                extra_criteria=extra,
                prompt=judge_prompt,
            )

    async def run(self, task: str, execute) -> str:
        """
        Run the agent on a task.

        Args:
            task    — the user's request
            execute — async fn(tool_call: ToolCall, task: str) -> dict | None
                      returns {"content": "..."} for each tool call

        Returns:
            The agent's final answer as a string.
        """
        state = AgentState(messages=[Message(role="system", content=self._system_prompt)])
        return await run(
            task=task,
            state=state,
            think=self._think,
            execute=execute,
            judge=self._judge,
            max_steps=self._max_steps,
        )
