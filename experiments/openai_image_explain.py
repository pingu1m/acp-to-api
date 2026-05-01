#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx",
#   "openai",
# ]
# ///

from __future__ import annotations

import argparse
import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from openai import OpenAI

DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/github/explore/main/topics/python/python.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload an online image to OpenAI-compatible chat and ask for an explanation."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434/api/v1/cursor/openai",
        help="OpenAI-compatible base URL from acp-to-api.",
    )
    parser.add_argument(
        "--api-key",
        default="dummy",
        help="API key sent to backend (dummy is fine for local acp-to-api).",
    )
    parser.add_argument(
        "--image-url",
        default=DEFAULT_IMAGE_URL,
        help="Public image URL to download and upload as data URL.",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="Model ID to use. Use 'auto' to select cursor/first available model.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Please explain this image in 3 concise bullet points. Describe visible objects, style, and likely context."
        ),
        help="Prompt to send with the uploaded image.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional markdown output file path. Defaults to experiments/results timestamped file.",
    )
    parser.add_argument(
        "--save-image-path",
        default="experiments/results/openai_image_explain_input",
        help="Where to save the downloaded test image (extension added automatically).",
    )
    return parser.parse_args()


def _guess_extension(content_type: str | None, image_url: str) -> tuple[str, str]:
    if content_type:
        mime_type = content_type.split(";", 1)[0].strip()
    else:
        guessed, _ = mimetypes.guess_type(image_url)
        mime_type = guessed or "image/png"
    extension = mimetypes.guess_extension(mime_type) or ".png"
    return mime_type, extension


def _select_model(client: OpenAI, preferred: str) -> tuple[str, list[str]]:
    model_ids = [m.id for m in client.models.list().data]
    if not model_ids:
        raise RuntimeError("No models returned by /models endpoint.")
    if preferred != "auto":
        if preferred not in model_ids:
            raise RuntimeError(f"Requested model '{preferred}' not found. Available: {model_ids}")
        return preferred, model_ids
    if "cursor" in model_ids:
        return "cursor", model_ids
    return model_ids[0], model_ids


def main() -> int:
    args = parse_args()
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    image_resp = httpx.get(args.image_url, timeout=30.0, follow_redirects=True)
    image_resp.raise_for_status()
    image_bytes = image_resp.content
    mime_type, extension = _guess_extension(image_resp.headers.get("content-type"), args.image_url)

    image_save_path = Path(args.save_image_path)
    if image_save_path.suffix:
        final_image_path = image_save_path
    else:
        final_image_path = image_save_path.with_suffix(extension)
    final_image_path.parent.mkdir(parents=True, exist_ok=True)
    final_image_path.write_bytes(image_bytes)

    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    model_id, discovered_models = _select_model(client, args.model)

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    explanation = (response.choices[0].message.content or "").strip()

    output_file = (
        Path(args.output_file)
        if args.output_file
        else results_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_openai_image_explain.md"
    )
    parsed = urlparse(args.image_url)
    markdown = (
        "# OpenAI Image Explain Experiment\n\n"
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- Base URL: `{args.base_url}`\n"
        f"- Model: `{model_id}`\n"
        f"- Image URL: `{args.image_url}`\n"
        f"- Image host: `{parsed.netloc}`\n"
        f"- Saved image: `{final_image_path}`\n\n"
        "## Prompt\n\n"
        f"{args.prompt}\n\n"
        "## Discovered Models\n\n"
        + "\n".join(f"- `{m}`" for m in discovered_models)
        + "\n\n## Explanation\n\n"
        + (explanation or "(empty response)")
        + "\n"
    )
    output_file.write_text(markdown, encoding="utf-8")

    print(f"Saved image: {final_image_path}")
    print(f"Saved report: {output_file}")
    print("\nExplanation:\n")
    print(explanation or "(empty response)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
