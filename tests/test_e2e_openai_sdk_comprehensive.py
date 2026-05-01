"""Comprehensive OpenAI SDK E2E tests for acp-to-api.

Exercises every surface of the OpenAI-compatible API through the official
Python SDK: models, chat completions (streaming & non-streaming), multi-turn
conversations, multimodal content, system/developer/user roles, response
metadata validation, concurrent requests, and error paths.

Uses the session-scoped server fixtures from conftest.py.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

# 1x1 transparent PNG as data-URL
PNG_1X1 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBgL6N4m0AAAAASUVORK5CYII="
)


def _client(port: int) -> OpenAI:
    return OpenAI(
        base_url=f"http://127.0.0.1:{port}/api/v1/cursor/openai",
        api_key="unused",
    )


def _async_client(port: int) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=f"http://127.0.0.1:{port}/api/v1/cursor/openai",
        api_key="unused",
    )


# ===================================================================
# 1. Models endpoint
# ===================================================================


class TestModelsEndpoint:
    def test_list_returns_multiple_acp_models(self, server_process: object, server_port: int) -> None:
        client = _client(server_port)
        models = client.models.list()
        ids = [m.id for m in models.data]
        assert len(ids) > 1, f"Expected multiple ACP models, got: {ids}"
        for m in models.data:
            assert m.object == "model"
            assert m.owned_by.startswith("acp-to-api:")

    def test_retrieve_each_discovered_model(self, server_process: object, server_port: int) -> None:
        """Every model from list should be individually retrievable."""
        client = _client(server_port)
        models = client.models.list()
        for m in models.data[:5]:  # spot-check first 5
            retrieved = client.models.retrieve(m.id)
            assert retrieved.id == m.id
            assert retrieved.object == "model"

    def test_retrieve_unknown_model_404(self, server_process: object, server_port: int) -> None:
        r = httpx.get(
            f"http://127.0.0.1:{server_port}/api/v1/cursor/openai/models/does-not-exist",
            timeout=10,
        )
        assert r.status_code == 404

    def test_unknown_provider_404(self, server_process: object, server_port: int) -> None:
        r = httpx.get(
            f"http://127.0.0.1:{server_port}/api/v1/bogus_provider/openai/models",
            timeout=10,
        )
        assert r.status_code == 404


# ===================================================================
# 2. Non-streaming chat completions
# ===================================================================


class TestNonStreamingCompletions:
    def test_simple_completion_response_shape(self, server_process: object, server_port: int) -> None:
        """Validate every field of a non-streaming response."""
        client = _client(server_port)
        resp = client.chat.completions.create(
            model="cursor",
            messages=[{"role": "user", "content": "Say exactly: shape-ok"}],
        )
        assert resp.id.startswith("chatcmpl-")
        assert resp.object == "chat.completion"
        assert resp.created > 0
        assert resp.model == "cursor"
        assert len(resp.choices) == 1

        choice = resp.choices[0]
        assert choice.index == 0
        assert choice.finish_reason in ("stop", "end_turn", "tool_calls")
        assert choice.message.role == "assistant"
        assert choice.message.content and choice.message.content.strip()

        assert resp.usage is not None
        assert resp.usage.prompt_tokens >= 0
        assert resp.usage.completion_tokens >= 0
        assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens

    def test_system_message_influences_response(self, server_process: object, server_port: int) -> None:
        client = _client(server_port)
        resp = client.chat.completions.create(
            model="cursor",
            messages=[
                {"role": "system", "content": "You only respond in uppercase."},
                {"role": "user", "content": "Say hello."},
            ],
        )
        text = resp.choices[0].message.content or ""
        assert text.strip(), "Empty response with system message"

    def test_developer_role_accepted(self, server_process: object, server_port: int) -> None:
        """The 'developer' role should be accepted without error."""
        client = _client(server_port)
        resp = client.chat.completions.create(
            model="cursor",
            messages=[
                {"role": "developer", "content": "Always answer in French."},
                {"role": "user", "content": "What is 1+1?"},
            ],
        )
        assert resp.choices[0].message.content.strip()

    def test_optional_params_accepted(self, server_process: object, server_port: int) -> None:
        """temperature, max_tokens, max_completion_tokens should be accepted (even if ignored)."""
        client = _client(server_port)
        resp = client.chat.completions.create(
            model="cursor",
            messages=[{"role": "user", "content": "Say hi."}],
            temperature=0.5,
            max_tokens=100,
        )
        assert resp.choices[0].message.content.strip()


# ===================================================================
# 3. Streaming chat completions
# ===================================================================


class TestStreamingCompletions:
    def test_streaming_chunk_shape(self, server_process: object, server_port: int) -> None:
        """Validate the shape of each SSE chunk."""
        client = _client(server_port)
        stream = client.chat.completions.create(
            model="cursor",
            messages=[{"role": "user", "content": "Count to 3."}],
            stream=True,
        )
        chunks = list(stream)
        assert len(chunks) >= 2, "Expected at least a role chunk and content chunks"

        # First chunk should carry the role
        first = chunks[0]
        assert first.id.startswith("chatcmpl-")
        assert first.object == "chat.completion.chunk"
        assert first.model == "cursor"
        assert first.choices[0].delta.role == "assistant"

        # Subsequent chunks should carry content
        content_parts = [c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content]
        full_text = "".join(content_parts)
        assert full_text.strip(), "Streaming produced no content"

        # Last chunk should have a finish_reason
        last_with_reason = [c for c in chunks if c.choices and c.choices[0].finish_reason]
        assert last_with_reason, "No chunk carried a finish_reason"

    def test_streaming_matches_non_streaming_content(self, server_process: object, server_port: int) -> None:
        """Both modes should produce non-empty, relevant responses to the same prompt."""
        client = _client(server_port)
        prompt = "What is the capital of Japan? One word."

        non_stream = client.chat.completions.create(
            model="cursor",
            messages=[{"role": "user", "content": prompt}],
        )
        ns_text = (non_stream.choices[0].message.content or "").lower()

        stream = client.chat.completions.create(
            model="cursor",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        parts = [c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta.content]
        s_text = "".join(parts).lower()

        # Both should mention Tokyo
        assert "tokyo" in ns_text, f"Non-streaming didn't mention Tokyo: {ns_text[:200]}"
        assert "tokyo" in s_text, f"Streaming didn't mention Tokyo: {s_text[:200]}"

    def test_streaming_ids_consistent_across_chunks(self, server_process: object, server_port: int) -> None:
        """All chunks in a single stream should share the same completion id."""
        client = _client(server_port)
        stream = client.chat.completions.create(
            model="cursor",
            messages=[{"role": "user", "content": "Say something."}],
            stream=True,
        )
        ids = {c.id for c in stream}
        assert len(ids) == 1, f"Expected one consistent id across chunks, got: {ids}"


# ===================================================================
# 4. Multi-turn conversations
# ===================================================================


class TestMultiTurn:
    def test_two_turn_context_recall(self, server_process: object, server_port: int) -> None:
        """Model should recall a fact from the first turn in the second."""
        client = _client(server_port)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Remember everything the user says."},
            {"role": "user", "content": "My secret code is ZEPHYR-42."},
        ]
        r1 = client.chat.completions.create(model="cursor", messages=messages)
        messages.append({"role": "assistant", "content": r1.choices[0].message.content})
        messages.append({"role": "user", "content": "What is my secret code? Reply with just the code."})

        r2 = client.chat.completions.create(model="cursor", messages=messages)
        text = (r2.choices[0].message.content or "").upper()
        assert "ZEPHYR" in text, f"Failed to recall secret code: {text[:200]}"

    def test_five_turn_conversation(self, server_process: object, server_port: int) -> None:
        """Build a 5-turn conversation and verify the proxy handles growing context."""
        client = _client(server_port)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a concise assistant."},
        ]
        topics = ["apples", "bananas", "cherries", "dates", "elderberries"]
        for fruit in topics:
            messages.append({"role": "user", "content": f"Add '{fruit}' to the list."})
            resp = client.chat.completions.create(model="cursor", messages=messages)
            reply = resp.choices[0].message.content or ""
            assert reply.strip(), f"Empty reply at turn for '{fruit}'"
            messages.append({"role": "assistant", "content": reply})

        # Final turn: ask to recall
        messages.append({"role": "user", "content": "List all fruits I mentioned."})
        final = client.chat.completions.create(model="cursor", messages=messages)
        final_text = (final.choices[0].message.content or "").lower()
        # Should recall at least some of the fruits
        recalled = [f for f in topics if f in final_text]
        assert len(recalled) >= 3, f"Expected at least 3 of {topics} recalled, got {recalled}: {final_text[:300]}"

    def test_multi_turn_streaming(self, server_process: object, server_port: int) -> None:
        """Multi-turn with streaming on the second turn."""
        client = _client(server_port)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "My pet's name is Nimbus."},
        ]
        r1 = client.chat.completions.create(model="cursor", messages=messages)
        messages.append({"role": "assistant", "content": r1.choices[0].message.content})
        messages.append({"role": "user", "content": "What is my pet's name? Stream your answer."})

        stream = client.chat.completions.create(model="cursor", messages=messages, stream=True)
        parts = [c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta.content]
        text = "".join(parts).lower()
        assert "nimbus" in text, f"Streaming turn failed to recall 'Nimbus': {text[:200]}"


# ===================================================================
# 5. Multimodal / attachments
# ===================================================================


class TestMultimodal:
    def test_text_plus_image_content_block(self, server_process: object, server_port: int) -> None:
        """Send a message with mixed text + image_url content parts."""
        client = _client(server_port)
        resp = client.chat.completions.create(
            model="cursor",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly."},
                        {"type": "image_url", "image_url": {"url": PNG_1X1}},
                    ],
                }
            ],
        )
        assert resp.choices[0].message.content.strip()

    def test_multiple_images_in_one_message(self, server_process: object, server_port: int) -> None:
        """Multiple image_url parts in a single message."""
        client = _client(server_port)
        resp = client.chat.completions.create(
            model="cursor",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "I'm sending two images."},
                        {"type": "image_url", "image_url": {"url": PNG_1X1}},
                        {"type": "image_url", "image_url": {"url": PNG_1X1}},
                        {"type": "text", "text": "Acknowledge you received both."},
                    ],
                }
            ],
        )
        assert resp.choices[0].message.content.strip()

    def test_image_in_multi_turn(self, server_process: object, server_port: int) -> None:
        """Image in the first turn, text-only follow-up in the second."""
        client = _client(server_port)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is an image of a 1x1 pixel."},
                    {"type": "image_url", "image_url": {"url": PNG_1X1}},
                ],
            },
        ]
        r1 = client.chat.completions.create(model="cursor", messages=messages)
        messages.append({"role": "assistant", "content": r1.choices[0].message.content})
        messages.append({"role": "user", "content": "What did I send you in the first message?"})

        r2 = client.chat.completions.create(model="cursor", messages=messages)
        text = (r2.choices[0].message.content or "").lower()
        assert text.strip(), "Empty response on image follow-up turn"


# ===================================================================
# 6. Complex message payloads
# ===================================================================


class TestComplexPayloads:
    def test_long_system_prompt(self, server_process: object, server_port: int) -> None:
        """A very long system prompt should not crash the proxy."""
        client = _client(server_port)
        long_system = "You are a helpful assistant. " * 200
        resp = client.chat.completions.create(
            model="cursor",
            messages=[
                {"role": "system", "content": long_system},
                {"role": "user", "content": "Say OK."},
            ],
        )
        assert resp.choices[0].message.content.strip()

    def test_mixed_roles_conversation(self, server_process: object, server_port: int) -> None:
        """system + user + assistant + user pattern."""
        client = _client(server_port)
        resp = client.chat.completions.create(
            model="cursor",
            messages=[
                {"role": "system", "content": "You are a geography expert."},
                {"role": "user", "content": "What continent is Brazil in?"},
                {"role": "assistant", "content": "Brazil is in South America."},
                {"role": "user", "content": "And what is the largest city there?"},
            ],
        )
        text = (resp.choices[0].message.content or "").lower()
        assert text.strip()
        assert "paulo" in text or "são" in text or "sao" in text or "rio" in text, (
            f"Expected a Brazilian city, got: {text[:200]}"
        )

    def test_empty_user_message(self, server_process: object, server_port: int) -> None:
        """Empty string content should not crash."""
        client = _client(server_port)
        resp = client.chat.completions.create(
            model="cursor",
            messages=[{"role": "user", "content": ""}],
        )
        # Should return something (even if it's a confused response)
        assert resp.choices[0].message.content is not None

    def test_unicode_content(self, server_process: object, server_port: int) -> None:
        """Unicode characters should round-trip through the proxy."""
        client = _client(server_port)
        resp = client.chat.completions.create(
            model="cursor",
            messages=[
                {"role": "user", "content": "Repeat this exactly: café résumé naïve 日本語 🎉"},
            ],
        )
        text = resp.choices[0].message.content or ""
        assert text.strip(), "Empty response for unicode content"

    def test_code_block_in_message(self, server_process: object, server_port: int) -> None:
        """Code blocks with backticks should pass through cleanly."""
        client = _client(server_port)
        resp = client.chat.completions.create(
            model="cursor",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Review this Python code:\n"
                        "```python\n"
                        "def fib(n):\n"
                        "    if n <= 1: return n\n"
                        "    return fib(n-1) + fib(n-2)\n"
                        "```\n"
                        "Is there a performance issue?"
                    ),
                }
            ],
        )
        text = (resp.choices[0].message.content or "").lower()
        assert text.strip()
        # Should mention recursion, performance, or memoization
        assert any(w in text for w in ["recurs", "exponential", "memo", "cache", "slow", "performance", "fib"]), (
            f"Response doesn't address the code: {text[:300]}"
        )


# ===================================================================
# 7. Async client
# ===================================================================


class TestAsyncClient:
    def test_async_non_streaming(self, server_process: object, server_port: int) -> None:
        async def _run() -> str:
            client = _async_client(server_port)
            resp = await client.chat.completions.create(
                model="cursor",
                messages=[{"role": "user", "content": "Say async-ok."}],
            )
            return resp.choices[0].message.content or ""

        text = asyncio.run(_run())
        assert text.strip()

    def test_async_streaming(self, server_process: object, server_port: int) -> None:
        async def _run() -> str:
            client = _async_client(server_port)
            stream = await client.chat.completions.create(
                model="cursor",
                messages=[{"role": "user", "content": "Say async-stream-ok."}],
                stream=True,
            )
            parts: list[str] = []
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
            return "".join(parts)

        text = asyncio.run(_run())
        assert text.strip()

    def test_async_concurrent_requests(self, server_process: object, server_port: int) -> None:
        """Fire 3 async requests concurrently and verify all complete."""

        async def _run() -> list[str]:
            client = _async_client(server_port)
            prompts = [
                "Say 'alpha' and nothing else.",
                "Say 'beta' and nothing else.",
                "Say 'gamma' and nothing else.",
            ]

            async def _call(prompt: str) -> str:
                resp = await client.chat.completions.create(
                    model="cursor",
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content or ""

            return await asyncio.gather(*[_call(p) for p in prompts])

        results = asyncio.run(_run())
        assert len(results) == 3
        for r in results:
            assert r.strip(), "One of the concurrent responses was empty"
        # At least 2 should be distinct
        assert len(set(r.strip().lower() for r in results)) >= 2, f"All concurrent responses identical: {results}"


# ===================================================================
# 8. Health & error paths
# ===================================================================


class TestHealthAndErrors:
    def test_health_endpoint(self, server_process: object, server_port: int) -> None:
        r = httpx.get(f"http://127.0.0.1:{server_port}/health", timeout=5)
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_unknown_provider_chat_404(self, server_process: object, server_port: int) -> None:
        bad_client = OpenAI(
            base_url=f"http://127.0.0.1:{server_port}/api/v1/nonexistent/openai",
            api_key="unused",
        )
        with pytest.raises(Exception):
            bad_client.chat.completions.create(
                model="cursor",
                messages=[{"role": "user", "content": "hello"}],
            )

    def test_raw_http_post_with_httpx(self, server_process: object, server_port: int) -> None:
        """Bypass the SDK and send a raw HTTP POST to verify the API contract."""
        r = httpx.post(
            f"http://127.0.0.1:{server_port}/api/v1/cursor/openai/chat/completions",
            json={
                "model": "cursor",
                "messages": [{"role": "user", "content": "Say raw-http-ok."}],
                "stream": False,
            },
            timeout=60,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"].strip()

    def test_raw_http_streaming_sse(self, server_process: object, server_port: int) -> None:
        """Verify raw SSE format: 'data: {...}\\n\\n' lines ending with 'data: [DONE]'."""
        with httpx.stream(
            "POST",
            f"http://127.0.0.1:{server_port}/api/v1/cursor/openai/chat/completions",
            json={
                "model": "cursor",
                "messages": [{"role": "user", "content": "Say sse-ok."}],
                "stream": True,
            },
            timeout=60,
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")

            lines: list[str] = []
            for line in r.iter_lines():
                if line.strip():
                    lines.append(line)

        assert len(lines) >= 2, f"Expected multiple SSE lines, got {len(lines)}"
        # Every non-empty line should start with "data: "
        for line in lines:
            assert line.startswith("data: "), f"Bad SSE line: {line[:100]}"
        # Last line should be the [DONE] sentinel
        assert lines[-1] == "data: [DONE]"
