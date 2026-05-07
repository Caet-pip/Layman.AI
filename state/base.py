"""
state/base.py

Bare bones AgentState. Tracks messages and the current task.
Extend this dataclass in your own project to add domain-specific fields.

Example extension (browser agent):
    from laymanai.state.base import AgentState
    from dataclasses import dataclass, field

    @dataclass
    class BrowserAgentState(AgentState):
        visited_urls: list[str] = field(default_factory=list)
        emitted_urls: set      = field(default_factory=set)
"""

from dataclasses import dataclass, field


@dataclass
class AgentState:
    # The current task string — set at the start of each run
    task: str = ""

    # Full message history sent to and received from the LLM
    # Format: [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}]
    messages: list[dict] = field(default_factory=list)
