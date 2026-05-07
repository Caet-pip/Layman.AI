"""
examples/local_file_agent.py

A minimal but complete agent built with laymanai.
Give it a directory and ask it anything about the files inside.

It can list files, read them, ask you clarifying questions,
and reason across multiple files to form an answer.

Run:
    export OPENAI_API_KEY=your_key
    python local_file_agent.py --dir ./my_folder

Or with Ollama:
    python local_file_agent.py --dir ./my_folder --backend ollama --model gemma4:4b
"""

import asyncio
import argparse
import json
import os
from pathlib import Path
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.loop import run
from core.judge import make_judge
from core.heal import heal
from state.base import AgentState
from tools.ask_human import execute as ask_human_execute, schema as ask_human_schema


# ── Tool schemas ───────────────────────────────────────────────────────────────

LIST_FILES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List all files in the target directory.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file in the target directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename to read (just the name, not the full path).",
                }
            },
            "required": ["filename"],
        },
    },
}

TOOLS = [LIST_FILES_SCHEMA, READ_FILE_SCHEMA, ask_human_schema]

SYSTEM_PROMPT = """You are a file assistant. You have access to a local directory of files.
You can list files, read them, and answer questions about their contents.
Always list files first before trying to read any.
If you need clarification from the user, use ask_human.
Be thorough — read all relevant files before answering."""


# ── Tool executor ──────────────────────────────────────────────────────────────

def make_execute(target_dir: Path):
    async def execute(tool_call, task: str):
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if name == "list_files":
            files = [f.name for f in target_dir.iterdir() if f.is_file()]
            if not files:
                return {"tool_call_id": tool_call.id, "content": "No files found in directory."}
            return {"tool_call_id": tool_call.id, "content": "\n".join(files)}

        if name == "read_file":
            filename = args.get("filename", "")
            path = target_dir / filename
            if not path.exists():
                return {"tool_call_id": tool_call.id, "content": f"File not found: {filename}"}
            if not path.is_relative_to(target_dir):
                return {"tool_call_id": tool_call.id, "content": "Access denied — file is outside target directory."}
            content = path.read_text(errors="replace")
            return {"tool_call_id": tool_call.id, "content": content}

        if name == "ask_human":
            return {**(await ask_human_execute(tool_call)), "tool_call_id": tool_call.id}

        return {"tool_call_id": tool_call.id, "content": f"Unknown tool: {name}"}

    return execute


# ── LLM think function ─────────────────────────────────────────────────────────

def make_think(client, model: str, tools: list):
    async def think(messages: list[dict]):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        return response.choices[0].message
    return think


# ── Entry point ────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory to browse")
    parser.add_argument("--backend", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model", help="Override model name")
    args = parser.parse_args()

    target_dir = Path(args.dir).resolve()
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: {target_dir} is not a valid directory.")
        return

    # LLM client setup
    if args.backend == "ollama":
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        model = args.model or "gemma4:4b"
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        model = args.model or "gpt-4.1"

    print(f"Local File Agent | {args.backend} / {model}")
    print(f"Directory: {target_dir}\n")

    # Wire up the agent
    state = AgentState(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
    )
    think   = make_think(client, model, TOOLS)
    execute = make_execute(target_dir)
    judge   = make_judge(client, model, max_rounds=2)

    task = input("What would you like to know about these files?\n> ").strip()
    if not task:
        return

    result = await run(
        task=task,
        state=state,
        think=think,
        execute=execute,
        judge=judge,
    )

    print(f"\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
