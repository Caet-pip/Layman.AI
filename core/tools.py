"""
tools/tools.py

Define tools once in a provider-neutral format.
Use bundle() to convert a list of tools to any provider's schema format.

Usage:
    from tools.tools import define_tool, bundle

    LIST_FILES = define_tool(
        name="list_files",
        description="List all files in the target directory.",
        properties={},
        required=[],
    )

    READ_FILE = define_tool(
        name="read_file",
        description="Read the contents of a file.",
        properties={"filename": {"type": "string", "description": "The file to read."}},
        required=["filename"],
    )

    TOOLS = bundle([LIST_FILES, READ_FILE], provider="openai")
    TOOLS = bundle([LIST_FILES, READ_FILE], provider="anthropic")
"""


def define_tool(name: str, description: str, properties: dict, required: list) -> dict:
    """Define a tool in a provider-neutral format."""
    return {
        "name": name,
        "description": description,
        "properties": properties,
        "required": required,
    }


def _to_openai(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {
                "type": "object",
                "properties": tool["properties"],
                "required": tool["required"],
            },
        },
    }


def _to_anthropic(tool: dict) -> dict:
    return {
        "name": tool["name"],
        "description": tool["description"],
        "input_schema": {
            "type": "object",
            "properties": tool["properties"],
            "required": tool["required"],
        },
    }


def _to_gemini(tool: dict) -> dict:
    # Produces a function declaration — gemini_think wraps these in {"function_declarations": [...]}
    return {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": {
            "type": "object",
            "properties": tool["properties"],
            "required": tool["required"],
        },
    }


_CONVERTERS = {
    "openai": _to_openai,
    "anthropic": _to_anthropic,
    "gemini": _to_gemini,
}


def bundle(tools: list, provider: str) -> list:
    """
    Convert a list of tool definitions to a provider's schema format.

    Args:
        tools    — list of dicts from define_tool()
        provider — "openai" or "anthropic"

    Returns:
        List of tool schemas ready to pass to the LLM client.
    """
    convert = _CONVERTERS.get(provider)
    if not convert:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_CONVERTERS)}")
    return [convert(t) for t in tools]
