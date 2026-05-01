#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx",
#   "langchain-openai",
#   "langgraph",
# ]
# ///

from __future__ import annotations

import argparse
import operator
import re
import sys
import textwrap
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


class ResearchState(TypedDict):
    question: str
    notes: Annotated[list[str], operator.add]
    final_report: str


def fetch_model_ids(base_url: str, timeout_seconds: float, backend: str) -> list[str]:
    url = f"{base_url.rstrip('/')}/models"
    try:
        response = httpx.get(url, timeout=timeout_seconds)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Failed to fetch models from {url}. "
            f"Ensure acp-to-api is running and the '{backend}' provider is configured."
        ) from exc

    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected models payload from {url}: {payload!r}")

    model_ids = [item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]
    if not model_ids:
        raise RuntimeError(
            "The models endpoint returned no model IDs. "
            f"{backend} ACP may still be starting, or model discovery failed."
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


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_text = item.get("text")
                if item_text:
                    text_parts.append(str(item_text))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts).strip()
    return str(content)


def build_research_graph(llm: ChatOpenAI) -> Any:
    def planner(state: ResearchState) -> dict[str, list[str]]:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a technical research planner. "
                        "Create a concise plan to evaluate whether Rust is a good fit for agentic coding."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Research question: {state['question']}\n\n"
                        "Return exactly 3 short bullet points focused on evaluation dimensions."
                    )
                ),
            ]
        )
        return {"notes": [f"Research plan:\n{_content_to_text(response.content)}"]}

    def evidence(state: ResearchState) -> dict[str, list[str]]:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a senior software architect. "
                        "Provide technical evidence grounded in language/runtime/tooling characteristics."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {state['question']}\n\n"
                        "Using the plan below, produce evidence with pros/cons for Rust agentic coding.\n"
                        f"{chr(10).join(state['notes'])}\n\n"
                        "Limit to 6 bullet points."
                    )
                ),
            ]
        )
        return {"notes": [f"Evidence:\n{_content_to_text(response.content)}"]}

    def synthesis(state: ResearchState) -> dict[str, str]:
        response = llm.invoke(
            [
                SystemMessage(
                    content=("You are writing a final recommendation memo. Be balanced, concise, and actionable.")
                ),
                HumanMessage(
                    content=(
                        f"Question: {state['question']}\n\n"
                        "Synthesize the research notes into:\n"
                        "1) short answer, 2) when Rust is a strong choice, "
                        "3) when Python is better, 4) a recommendation for a first prototype.\n\n"
                        "Research notes:\n"
                        f"{chr(10).join(state['notes'])}"
                    )
                ),
            ]
        )
        return {"final_report": _content_to_text(response.content)}

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner)
    graph.add_node("evidence", evidence)
    graph.add_node("synthesis", synthesis)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "evidence")
    graph.add_edge("evidence", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()


def run_research(question: str, base_url: str, api_key: str, model_id: str) -> tuple[str, float]:
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_id,
        temperature=0,
        streaming=False,
    )
    graph = build_research_graph(llm)
    start = time.perf_counter()
    result = graph.invoke({"question": question, "notes": [], "final_report": ""})
    elapsed_seconds = time.perf_counter() - start
    return result["final_report"], elapsed_seconds


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "research"


def _resolve_base_url(explicit_base_url: str | None, backend: str) -> str:
    if explicit_base_url:
        return explicit_base_url
    return f"http://127.0.0.1:11434/api/v1/{backend}/openai"


def build_markdown_report(
    *,
    question: str,
    backend: str,
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
        f"# LangGraph Research Report\n\n"
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- Backend: `{backend}`\n"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LangGraph research pipeline against acp-to-api models.")
    parser.add_argument(
        "--question",
        default="Is Rust a good choice for agentic coding?",
        help="Research question to run through the pipeline.",
    )
    parser.add_argument(
        "--backend",
        default="cursor",
        choices=["cursor", "kiro"],
        help="Configured acp-to-api provider backend to use.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=("Optional OpenAI-compatible base URL. Defaults to http://127.0.0.1:11434/api/v1/<backend>/openai."),
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
    backend = args.backend
    base_url = _resolve_base_url(args.base_url, backend)
    try:
        model_ids = fetch_model_ids(base_url, args.timeout_seconds, backend)

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
        print(f"- Backend: {backend}")
        print(f"- Base URL: {base_url}")
        print(f"- Sonnet: {sonnet_model}")
        print(f"- Opus:   {opus_model}")

        if interactive_selection and not _confirm_selected_models(sonnet_model, opus_model):
            print("Cancelled by user before running the research pipeline.")
            return 0

        print("\nRunning research pipeline...")
        sonnet_report, sonnet_time = run_research(args.question, base_url, args.api_key, sonnet_model)
        opus_report, opus_time = run_research(args.question, base_url, args.api_key, opus_model)

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
                backend=backend,
                base_url=base_url,
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
                "Tip: start the proxy first, for example:\n  uv run acp-to-api serve --config acp-to-api.toml",
                file=sys.stderr,
            )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected failure while running research pipeline: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
