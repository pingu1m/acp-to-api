#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx",
#   "pydantic-ai",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import textwrap
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def fetch_model_ids(base_url: str, timeout_seconds: float) -> list[str]:
    url = f"{base_url.rstrip('/')}/models"
    try:
        response = httpx.get(url, timeout=timeout_seconds)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Failed to fetch models from {url}. Ensure acp-to-api is running and the Cursor provider is configured."
        ) from exc

    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected models payload from {url}: {payload!r}")

    model_ids = [item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]
    if not model_ids:
        raise RuntimeError(
            "The models endpoint returned no model IDs. Cursor ACP may still be starting, or model discovery failed."
        )
    return model_ids


def _should_show_startup_tip(message: str) -> bool:
    lowered = message.lower()
    return "failed to fetch models" in lowered or "models endpoint returned no model ids" in lowered


def select_model(model_ids: Sequence[str], family: str) -> str:
    family_lower = family.lower()
    version_pattern = re.compile(r"(?<!\d)4[.\-_ ]?6(?!\d)")
    scored: list[tuple[int, str]] = []

    for model_id in model_ids:
        lowered = model_id.lower()
        if family_lower not in lowered:
            continue
        score = 0
        if version_pattern.search(lowered):
            score += 100
        if "claude" in lowered:
            score += 20
        if "thinking" not in lowered:
            score += 5
        score += max(0, 15 - len(model_id) // 8)
        scored.append((score, model_id))

    if not scored:
        raise RuntimeError(
            f"Could not find any '{family}' model in discovered IDs:\n"
            + "\n".join(f"- {model_id}" for model_id in model_ids)
        )

    scored.sort(reverse=True)
    best_score, best_model = scored[0]
    if best_score < 100:
        raise RuntimeError(
            f"Found '{family}' models but none clearly matched version 4.6.\n"
            "Discovered IDs:\n" + "\n".join(f"- {model_id}" for model_id in model_ids)
        )
    return best_model


def _validate_model_choice(model_id: str, model_ids: Sequence[str], arg_name: str) -> str:
    if model_id not in model_ids:
        raise RuntimeError(
            f"{arg_name} value '{model_id}' was not found in discovered models.\n"
            "Discovered IDs:\n" + "\n".join(f"- {m}" for m in model_ids)
        )
    return model_id


def _prompt_model_choice(label: str, model_ids: Sequence[str], default_model: str) -> str:
    print(f"\nChoose {label} model:")
    for idx, model_id in enumerate(model_ids, start=1):
        marker = " (default)" if model_id == default_model else ""
        print(f"  {idx:>2}. {model_id}{marker}")

    while True:
        raw = input(f"{label} model number or exact ID [{default_model}]: ").strip()
        if not raw:
            return default_model
        if raw.isdigit():
            selected_index = int(raw)
            if 1 <= selected_index <= len(model_ids):
                return model_ids[selected_index - 1]
        if raw in model_ids:
            return raw
        print("Invalid selection. Enter a listed number or exact model ID.")


def _confirm_selected_models(sonnet_model: str, opus_model: str) -> bool:
    raw = (
        input(f"\nProceed with these models?\n  - Sonnet: {sonnet_model}\n  - Opus:   {opus_model}\nContinue [Y/n]: ")
        .strip()
        .lower()
    )
    return raw in {"", "y", "yes"}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "research"


def build_markdown_report(
    *,
    question: str,
    base_url: str,
    discovered_models: Sequence[str],
    sonnet_model: str,
    sonnet_time: float,
    sonnet_report: str,
    opus_model: str,
    opus_time: float,
    opus_report: str,
) -> str:
    discovered_block = "\n".join(f"- `{model}`" for model in discovered_models)
    return (
        f"# PydanticAI Research Report\n\n"
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- Base URL: `{base_url}`\n"
        f"- Question: {question}\n\n"
        f"## Discovered Models\n\n"
        f"{discovered_block}\n\n"
        f"## Sonnet 4.6 Result\n\n"
        f"- Model: `{sonnet_model}`\n"
        f"- Elapsed: `{sonnet_time:.2f}s`\n\n"
        f"{sonnet_report.strip()}\n\n"
        f"## Opus 4.6 Result\n\n"
        f"- Model: `{opus_model}`\n"
        f"- Elapsed: `{opus_time:.2f}s`\n\n"
        f"{opus_report.strip()}\n"
    )


def write_report(
    markdown: str,
    *,
    output_dir: str,
    output_file: str | None,
    question: str,
) -> Path:
    if output_file:
        path = Path(output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        question_slug = _slugify(question)[:60]
        path = Path(output_dir) / f"{timestamp}_{question_slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def _make_agent(model_id: str, base_url: str, api_key: str, system_prompt: str) -> Agent[None, str]:
    model = OpenAIChatModel(
        model_id,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
        ),
    )
    return Agent(
        model,
        system_prompt=system_prompt,
        output_type=str,
    )


async def run_research(question: str, base_url: str, api_key: str, model_id: str) -> tuple[str, float]:
    planner_agent = _make_agent(
        model_id,
        base_url,
        api_key,
        "You are a technical research planner. Create a concise plan.",
    )
    evidence_agent = _make_agent(
        model_id,
        base_url,
        api_key,
        "You are a senior software architect. Provide clear evidence with pros and cons.",
    )
    synthesis_agent = _make_agent(
        model_id,
        base_url,
        api_key,
        "You write final recommendation memos. Be balanced, concise, and actionable.",
    )

    start = time.perf_counter()

    plan_result = await planner_agent.run(
        f"Research question: {question}\n\nReturn exactly 3 short bullet points with evaluation dimensions."
    )
    plan_text = plan_result.output

    evidence_result = await evidence_agent.run(
        f"Question: {question}\n\n"
        f"Plan:\n{plan_text}\n\n"
        "Produce up to 6 bullet points with technical evidence (pros and cons)."
    )
    evidence_text = evidence_result.output

    synthesis_result = await synthesis_agent.run(
        f"Question: {question}\n\n"
        "Synthesize the notes into:\n"
        "1) short answer, 2) when Rust is a strong choice, "
        "3) when Python is better, 4) recommendation for a first prototype.\n\n"
        f"Research plan:\n{plan_text}\n\n"
        f"Evidence:\n{evidence_text}"
    )
    final_report = synthesis_result.output

    elapsed_seconds = time.perf_counter() - start
    return final_report, elapsed_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a PydanticAI research pipeline against Cursor-backed acp-to-api models."
    )
    parser.add_argument(
        "--question",
        default="Is Rust a good choice for agentic coding?",
        help="Research question to run through the pipeline.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/api/v1/cursor/openai",
        help="OpenAI-compatible base URL exposed by acp-to-api.",
    )
    parser.add_argument(
        "--api-key",
        default="dummy",
        help="API key passed to the OpenAI-compatible client (can be a dummy value).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="HTTP timeout used when fetching available models.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results",
        help="Directory used for auto-generated markdown reports.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional explicit markdown output file path.",
    )
    parser.add_argument(
        "--no-save-report",
        action="store_true",
        help="Disable writing the markdown report to disk.",
    )
    parser.add_argument(
        "--sonnet-model",
        default=None,
        help="Optional explicit model ID to use for the Sonnet run.",
    )
    parser.add_argument(
        "--opus-model",
        default=None,
        help="Optional explicit model ID to use for the Opus run.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip interactive model selection prompts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model_ids = fetch_model_ids(args.base_url, args.timeout_seconds)

        sonnet_default = select_model(model_ids, "sonnet")
        opus_default = select_model(model_ids, "opus")

        print("Discovered model IDs:")
        for model_id in model_ids:
            print(f"- {model_id}")

        interactive_selection = (not args.non_interactive) and sys.stdin.isatty()
        if args.sonnet_model:
            sonnet_model = _validate_model_choice(args.sonnet_model, model_ids, "--sonnet-model")
        elif interactive_selection:
            sonnet_model = _prompt_model_choice("Sonnet", model_ids, sonnet_default)
        else:
            sonnet_model = sonnet_default

        if args.opus_model:
            opus_model = _validate_model_choice(args.opus_model, model_ids, "--opus-model")
        elif interactive_selection:
            opus_model = _prompt_model_choice("Opus", model_ids, opus_default)
        else:
            opus_model = opus_default

        print("\nSelected models:")
        print(f"- Sonnet: {sonnet_model}")
        print(f"- Opus:   {opus_model}")

        if interactive_selection and not _confirm_selected_models(sonnet_model, opus_model):
            print("Cancelled by user before running the research pipeline.")
            return 0

        print("\nRunning PydanticAI research pipeline...")
        sonnet_report, sonnet_time = asyncio.run(run_research(args.question, args.base_url, args.api_key, sonnet_model))
        opus_report, opus_time = asyncio.run(run_research(args.question, args.base_url, args.api_key, opus_model))

        print("\n" + "=" * 80)
        print(f"Research question: {args.question}")
        print("=" * 80)

        print("\n[Sonnet 4.6 Result]")
        print(f"Model: {sonnet_model}")
        print(f"Elapsed: {sonnet_time:.2f}s")
        print(textwrap.dedent(sonnet_report).strip())

        print("\n" + "-" * 80)
        print("[Opus 4.6 Result]")
        print(f"Model: {opus_model}")
        print(f"Elapsed: {opus_time:.2f}s")
        print(textwrap.dedent(opus_report).strip())

        if not args.no_save_report:
            report_markdown = build_markdown_report(
                question=args.question,
                base_url=args.base_url,
                discovered_models=model_ids,
                sonnet_model=sonnet_model,
                sonnet_time=sonnet_time,
                sonnet_report=sonnet_report,
                opus_model=opus_model,
                opus_time=opus_time,
                opus_report=opus_report,
            )
            report_path = write_report(
                report_markdown,
                output_dir=args.output_dir,
                output_file=args.output_file,
                question=args.question,
            )
            print(f"\nSaved markdown report to: {report_path}")
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if _should_show_startup_tip(str(exc)):
            print(
                "Tip: start the proxy first, for example:\n"
                '  uv run acp-to-api serve --provider \'{"name":"cursor","command":"agent","args":["acp"]}\'',
                file=sys.stderr,
            )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected failure while running research pipeline: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
