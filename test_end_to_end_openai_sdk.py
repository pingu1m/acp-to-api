from __future__ import annotations

import argparse
import sys

import httpx
from openai import OpenAI


def run(base_url: str) -> int:
    # 1) Health check on proxy root.
    health_url = base_url.rsplit("/api/v1/", 1)[0] + "/health"
    health = httpx.get(health_url, timeout=10.0)
    health.raise_for_status()
    print("health:", health.json())

    # 2) OpenAI-compatible client against provider endpoint.
    client = OpenAI(base_url=base_url, api_key="dummy")

    # 3) Models endpoint.
    models = client.models.list()
    model_ids = [item.id for item in models.data]
    print("models:", model_ids)
    assert "cursor" in model_ids, "cursor model not exposed by proxy"

    # 4) Non-streaming completion.
    non_stream_resp = client.chat.completions.create(
        model="cursor",
        messages=[{"role": "user", "content": "Reply with exactly: config-e2e-non-stream-ok"}],
    )
    non_stream_text = (non_stream_resp.choices[0].message.content or "").strip()
    print("non_stream_text:", non_stream_text)
    assert non_stream_text, "non-stream response is empty"

    # 5) Streaming completion.
    stream = client.chat.completions.create(
        model="cursor",
        messages=[{"role": "user", "content": "Reply with exactly: config-e2e-stream-ok"}],
        stream=True,
    )
    parts: list[str] = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    stream_text = "".join(parts).strip()
    print("stream_text:", stream_text)
    assert stream_text, "stream response is empty"

    print("E2E OK: binary + OpenAI endpoint + SDK communication succeeded.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end test for acp-to-api using OpenAI SDK")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:18134/api/v1/cursor/openai",
        help="OpenAI-compatible base URL for the provider endpoint",
    )
    args = parser.parse_args()
    try:
        return run(args.base_url)
    except Exception as exc:  # pragma: no cover
        print(f"E2E FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
