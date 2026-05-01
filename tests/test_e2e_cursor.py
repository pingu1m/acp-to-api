from __future__ import annotations

from openai import OpenAI


def _client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="dummy")


def test_models_endpoint(server_process: object, server_port: int) -> None:
    client = _client(f"http://127.0.0.1:{server_port}/api/v1/cursor/openai")
    models = client.models.list()
    ids = [item.id for item in models.data]
    # Should discover real ACP models, not just a hardcoded "cursor"
    assert len(ids) > 1, f"Expected multiple models from ACP, got: {ids}"
    # Every model should have the required OpenAI fields
    for item in models.data:
        assert item.id, "Model id must not be empty"
        assert item.object == "model"
        assert item.owned_by.startswith("acp-to-api:")


def test_models_retrieve(server_process: object, server_port: int) -> None:
    client = _client(f"http://127.0.0.1:{server_port}/api/v1/cursor/openai")
    models = client.models.list()
    assert models.data, "No models returned"
    # Retrieve the first model by id
    first_id = models.data[0].id
    model = client.models.retrieve(first_id)
    assert model.id == first_id
    assert model.object == "model"


def test_models_retrieve_unknown_returns_404(server_process: object, server_port: int) -> None:
    import httpx

    r = httpx.get(
        f"http://127.0.0.1:{server_port}/api/v1/cursor/openai/models/nonexistent-model",
        timeout=10,
    )
    assert r.status_code == 404


def test_non_streaming_chat_completion(server_process: object, server_port: int) -> None:
    client = _client(f"http://127.0.0.1:{server_port}/api/v1/cursor/openai")
    response = client.chat.completions.create(
        model="cursor",
        messages=[{"role": "user", "content": "Reply with exactly: hello-from-acp"}],
    )
    text = response.choices[0].message.content or ""
    assert text.strip() != ""


def test_streaming_chat_completion(server_process: object, server_port: int) -> None:
    client = _client(f"http://127.0.0.1:{server_port}/api/v1/cursor/openai")
    stream = client.chat.completions.create(
        model="cursor",
        messages=[{"role": "user", "content": "Reply with exactly: streaming-from-acp"}],
        stream=True,
    )

    chunks: list[str] = []
    for part in stream:
        if not part.choices:
            continue
        delta = part.choices[0].delta
        if delta and delta.content:
            chunks.append(delta.content)

    assert "".join(chunks).strip() != ""
