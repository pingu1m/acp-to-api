#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "claude-agent-sdk",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run a Claude Code SDK experiment against acp-to-api to generate a Hello World Python example.")
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434/api/v1/cursor/anthropic",
        help="Anthropic-compatible base URL served by acp-to-api.",
    )
    parser.add_argument(
        "--api-key",
        default="dummy",
        help="API key sent to the backend (acp-to-api ignores value by default).",
    )
    parser.add_argument(
        "--model",
        default="sonnet",
        help="Claude model alias or full model name to request.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Maximum Claude Code SDK turns.",
    )
    parser.add_argument(
        "--workspace",
        default="experiments/claude_code_sdk_workspace",
        help="Workspace directory passed to the Claude Code SDK.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Generate a minimal Python hello world example. "
            "Return only a single Python code block that prints exactly "
            "'Hello, world!'."
        ),
        help="Prompt sent to Claude Code SDK.",
    )
    parser.add_argument(
        "--code-output-file",
        default="experiments/results/claude_sdk_hello_world.py",
        help="Where to write the generated Python code.",
    )
    parser.add_argument(
        "--report-output-file",
        default=None,
        help="Optional markdown report output path. Defaults to timestamped file in experiments/results.",
    )
    return parser.parse_args()


def _extract_python_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    stripped = text.strip()
    if stripped:
        return stripped + ("\n" if not stripped.endswith("\n") else "")
    return "print('Hello, world!')\n"


def _build_report(
    *,
    prompt: str,
    base_url: str,
    model: str,
    elapsed_seconds: float,
    code_output_file: Path,
    raw_response: str,
) -> str:
    return (
        "# Claude Code SDK Hello World Experiment\n\n"
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- Base URL: `{base_url}`\n"
        f"- Model: `{model}`\n"
        f"- Elapsed: `{elapsed_seconds:.2f}s`\n"
        f"- Code output: `{code_output_file}`\n\n"
        "## Prompt\n\n"
        f"{prompt}\n\n"
        "## Raw Response\n\n"
        f"{raw_response.strip()}\n"
    )


async def run_experiment(args: argparse.Namespace) -> tuple[str, float]:
    from claude_agent_sdk import ClaudeAgentOptions, query

    workspace_path = Path(args.workspace).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)

    os.environ["ANTHROPIC_API_KEY"] = args.api_key
    os.environ["ANTHROPIC_BASE_URL"] = args.base_url

    options = ClaudeAgentOptions(
        cwd=str(workspace_path),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
        permission_mode="bypassPermissions",
        max_turns=args.max_turns,
        model=args.model,
    )

    start = time.perf_counter()
    text_parts: list[str] = []

    async for message in query(prompt=args.prompt, options=options):
        if hasattr(message, "content") and isinstance(message.content, list):
            for block in message.content:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str) and block_text.strip():
                    text_parts.append(block_text)

    elapsed = time.perf_counter() - start
    return "\n\n".join(text_parts).strip(), elapsed


async def main() -> int:
    args = parse_args()
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_response, elapsed_seconds = await run_experiment(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1

    code = _extract_python_code(raw_response)
    code_output_file = Path(args.code_output_file)
    code_output_file.parent.mkdir(parents=True, exist_ok=True)
    code_output_file.write_text(code, encoding="utf-8")

    report_output_file = (
        Path(args.report_output_file)
        if args.report_output_file
        else results_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_claude_sdk_hello_world.md"
    )
    report = _build_report(
        prompt=args.prompt,
        base_url=args.base_url,
        model=args.model,
        elapsed_seconds=elapsed_seconds,
        code_output_file=code_output_file,
        raw_response=raw_response or "(no text response)",
    )
    report_output_file.write_text(report, encoding="utf-8")

    print(f"Code written to: {code_output_file}")
    print(f"Report written to: {report_output_file}")
    print("\nGenerated code:\n")
    print(code, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
