"""
core/types.py

Shared data types used across the harness.
All providers, the loop, heal, and state work with these shapes.
"""

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list = field(default_factory=list)  # list[ToolCall]


@dataclass
class Message:
    role: str                                        # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list = field(default_factory=list)   # list[ToolCall], populated for role="assistant"
    tool_call_id: str = ""                           # populated for role="tool"
    tool_name: str = ""                              # populated for role="tool"
