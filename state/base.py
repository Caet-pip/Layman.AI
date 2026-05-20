"""
state/base.py

Bare bones AgentState. Tracks the conversation as a list of Message objects.
Extend this dataclass in your own project to add domain-specific fields.

Example extension (browser agent):
    from state.base import AgentState
    from dataclasses import dataclass, field

    @dataclass
    class BrowserAgentState(AgentState):
        visited_urls: list[str] = field(default_factory=list)
        emitted_urls: set      = field(default_factory=set)
"""

from dataclasses import dataclass, field
from core.types import Message


@dataclass
class AgentState:
    task: str = ""
    messages: list = field(default_factory=list)  # list[Message]
