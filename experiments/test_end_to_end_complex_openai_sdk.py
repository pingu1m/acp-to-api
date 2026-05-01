from __future__ import annotations

import argparse
import sys

import httpx
from openai import OpenAI

# 1x1 PNG (transparent), encoded as data URL for attachment-style image input.
PNG_1X1_DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBgL6N4m0AAAAASUVORK5CYII="
)


def run(base_url: str) -> int:
    health_url = base_url.rsplit("/api/v1/", 1)[0] + "/health"
    health = httpx.get(health_url, timeout=10.0)
    health.raise_for_status()
    print("health:", health.json())

    client = OpenAI(base_url=base_url, api_key="dummy")
    models = client.models.list()
    model_ids = [item.id for item in models.data]
    print("models:", model_ids)
    assert "cursor" in model_ids, "cursor model not exposed by proxy"

    complex_messages = [
        {
            "role": "system",
            "content": "You are validating a complex API payload roundtrip.",
        },
        {
            "role": "user",
            "content": "Context A: My project is called Phoenix.",
        },
        {
            "role": "assistant",
            "content": "Acknowledged. Project name is Phoenix.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Context B: this message includes an image attachment."},
                {"type": "image_url", "image_url": {"url": PNG_1X1_DATA_URL}},
                {"type": "text", "text": "Please keep the project name and mention that an image was attached."},
            ],
        },
        {
            "role": "user",
            "content": (
                "Return exactly one line in this template: "
                "project=<name>;attachment_seen=<yes_or_no>;token=complex-e2e-ok"
            ),
        },
    ]

    non_stream_resp = client.chat.completions.create(
        model="cursor",
        messages=complex_messages,
    )
    non_stream_text = (non_stream_resp.choices[0].message.content or "").strip()
    print("complex_non_stream_text:", non_stream_text)
    assert non_stream_text, "complex non-stream response is empty"

    # Validate key markers so we know context + attachment style content got through.
    assert "project=" in non_stream_text.lower()
    assert "attachment" in non_stream_text.lower()
    assert "complex-e2e-ok" in non_stream_text.lower()

    stream = client.chat.completions.create(
        model="cursor",
        messages=complex_messages,
        stream=True,
    )
    parts: list[str] = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    stream_text = "".join(parts).strip()
    print("complex_stream_text:", stream_text)
    assert stream_text, "complex stream response is empty"
    assert "complex-e2e-ok" in stream_text.lower()

    print("COMPLEX E2E OK: rich messages + image attachment payload worked.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Complex end-to-end test for acp-to-api using OpenAI SDK and image payloads"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:18134/api/v1/cursor/openai",
        help="OpenAI-compatible base URL for the provider endpoint",
    )
    args = parser.parse_args()

    try:
        return run(args.base_url)
    except Exception as exc:  # pragma: no cover
        print(f"COMPLEX E2E FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
