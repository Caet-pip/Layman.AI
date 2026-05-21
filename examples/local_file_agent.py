"""
examples/local_file_agent.py

A minimal but complete agent built with laymanai.
Give it a directory and ask it anything about the files inside.

Run:
    export OPENAI_API_KEY=your_key
    python examples/local_file_agent.py --dir ./my_folder

    export GEMINI_API_KEY=your_key
    python examples/local_file_agent.py --dir ./my_folder --backend gemini

    python examples/local_file_agent.py --dir ./my_folder --backend ollama --model gemma4:4b
"""

import asyncio
import argparse
import os
from pathlib import Path
from openai import AsyncOpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent import Agent
from core.providers import OLLAMA_BASE_URL
from core.tools import define_tool


# ── Tool definitions ───────────────────────────────────────────────────────────

LIST_FILES = define_tool(
    name="list_files",
    description="List all files in the target directory.",
    properties={},
    required=[],
)

READ_FILE = define_tool(
    name="read_file",
    description="Read the contents of a file in the target directory.",
    properties={
        "filename": {
            "type": "string",
            "description": "The filename to read (just the name, not the full path).",
        }
    },
    required=["filename"],
)

ASK_HUMAN = define_tool(
    name="ask_human",
    description=(
        "Ask the human user for input, clarification, or a decision. "
        "Use this when you are genuinely blocked and cannot proceed without human input. "
        "Do NOT use this to confirm routine steps or ask if you should continue."
    ),
    properties={
        "question": {"type": "string", "description": "The question to ask the human."}
    },
    required=["question"],
)

SYSTEM_PROMPT = """You are a file assistant. You have access to a local directory of files.
You can list files, read them, and answer questions about their contents.
Always list files first before trying to read any.
If you need clarification from the user, use ask_human.
Be thorough — read all relevant files before answering."""


# ── Tool executor ──────────────────────────────────────────────────────────────

def make_execute(target_dir: Path):
    async def execute(tool_call, task: str):
        name = tool_call.name
        args = tool_call.arguments

        if name == "list_files":
            files = [f.name for f in target_dir.iterdir() if f.is_file()]
            if not files:
                return {"content": "No files found in directory."}
            return {"content": "\n".join(files)}

        if name == "read_file":
            filename = args.get("filename", "")
            path = target_dir / filename
            if not path.is_relative_to(target_dir):
                return {"content": "Access denied — file is outside target directory."}
            if not path.exists():
                return {"content": f"File not found: {filename}"}
            return {"content": path.read_text(errors="replace")}

        if name == "ask_human":
            question = args.get("question", "")
            print(f"\n[ask_human] {question}")
            answer = input("Your answer: ").strip()
            print(f"[human] {answer}")
            return {"content": answer}

        return {"content": f"Unknown tool: {name}"}

    return execute


# ── Entry point ────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",     required=True, help="Directory to browse")
    parser.add_argument("--backend", choices=["openai", "ollama", "gemini"], default="openai")
    parser.add_argument("--model",   help="Override model name")
    args = parser.parse_args()

    target_dir = Path(args.dir).resolve()
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: {target_dir} is not a valid directory.")
        return

    if args.backend == "gemini":
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
        model  = args.model or "gemini-3.5-flash"
    elif args.backend == "ollama":
        client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        model  = args.model or "gemma4:4b"
    else:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        model  = args.model or "gpt-5.5-instant"

    agent = Agent(
        client=client,
        model=model,
        provider=args.backend,
        tools=[LIST_FILES, READ_FILE, ASK_HUMAN],
        system_prompt=SYSTEM_PROMPT,
        judge="Answer must reference specific filenames found in the directory.",
    )

    print(f"Local File Agent | {args.backend} / {model}")
    print(f"Directory: {target_dir}\n")

    task = input("What would you like to know about these files?\n> ").strip()
    if not task:
        return

    result = await agent.run(task=task, execute=make_execute(target_dir))
    print(f"\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
