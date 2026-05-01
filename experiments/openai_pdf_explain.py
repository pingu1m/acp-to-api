#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx",
#   "openai",
#   "pypdf",
# ]
# ///

from __future__ import annotations

import argparse
import base64
import io
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from openai import OpenAI
from pypdf import PdfReader

DEFAULT_PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a public PDF to the OpenAI endpoint and ask for an explanation."
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
        "--pdf-url",
        default=DEFAULT_PDF_URL,
        help="Public PDF URL to download and include in the request.",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="Model ID to use. Use 'auto' to select cursor/first available model.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Please explain this PDF in 3 concise bullet points: "
            "what it appears to be, the key content, and likely purpose."
        ),
        help="Prompt to send with the uploaded PDF.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=3500,
        help="Maximum extracted PDF text characters included in prompt context.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional markdown output path. Defaults to experiments/results timestamped file.",
    )
    parser.add_argument(
        "--save-pdf-path",
        default="experiments/results/openai_pdf_explain_input.pdf",
        help="Where to save the downloaded PDF.",
    )
    return parser.parse_args()


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


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            chunks.append(text)
    return "\n\n".join(chunks).strip()


def main() -> int:
    args = parse_args()
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    pdf_resp = httpx.get(args.pdf_url, timeout=30.0, follow_redirects=True)
    pdf_resp.raise_for_status()
    pdf_bytes = pdf_resp.content

    pdf_save_path = Path(args.save_pdf_path)
    pdf_save_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_save_path.write_bytes(pdf_bytes)

    extracted_text = _extract_pdf_text(pdf_bytes)
    text_excerpt = extracted_text[: args.max_text_chars]
    pdf_data_url = f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode('ascii')}"

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    model_id, discovered_models = _select_model(client, args.model)

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{args.prompt}\n\n"
                            "Below is an extracted text excerpt from the same PDF "
                            "(for context if file parsing is limited):\n\n"
                            f"{text_excerpt or '(no extractable text found)'}"
                        ),
                    },
                    # acp-to-api currently supports image_url-style multimodal slots.
                    # We attach the PDF as a data URL payload for upload-style testing.
                    {"type": "image_url", "image_url": {"url": pdf_data_url}},
                ],
            }
        ],
    )
    explanation = (response.choices[0].message.content or "").strip()

    output_file = (
        Path(args.output_file)
        if args.output_file
        else results_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_openai_pdf_explain.md"
    )
    parsed = urlparse(args.pdf_url)
    markdown = (
        "# OpenAI PDF Explain Experiment\n\n"
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- Base URL: `{args.base_url}`\n"
        f"- Model: `{model_id}`\n"
        f"- PDF URL: `{args.pdf_url}`\n"
        f"- PDF host: `{parsed.netloc}`\n"
        f"- Saved PDF: `{pdf_save_path}`\n\n"
        "## Prompt\n\n"
        f"{args.prompt}\n\n"
        "## Extracted Text Excerpt\n\n"
        f"{(text_excerpt or '(no extractable text found)').strip()}\n\n"
        "## Discovered Models\n\n"
        + "\n".join(f"- `{m}`" for m in discovered_models)
        + "\n\n## Explanation\n\n"
        + (explanation or "(empty response)")
        + "\n"
    )
    output_file.write_text(markdown, encoding="utf-8")

    print(f"Saved PDF: {pdf_save_path}")
    print(f"Saved report: {output_file}")
    print("\nExplanation:\n")
    print(explanation or "(empty response)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
