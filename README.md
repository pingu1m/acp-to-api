# acp-to-api

`acp-to-api` is a multi-provider proxy CLI that exposes OpenAI-compatible REST APIs backed by ACP (Agent Client Protocol) subprocess providers.

For each configured provider, the server exposes:

- `GET /api/v1/<provider>/openai/models`
- `POST /api/v1/<provider>/openai/chat/completions`

The first provider implemented is Cursor CLI ACP (`agent acp`).

## Runtime resilience

The Cursor ACP provider monitors its subprocess health and automatically
restarts the ACP connection if the subprocess exits unexpectedly.

## Limitations

- **OpenAI tuning parameters** (`temperature`, `max_tokens`, `max_completion_tokens`) are accepted in the request schema for API compatibility, but are **not forwarded** to the ACP agent — the underlying protocol does not expose these controls.
- **Token usage** in responses is estimated from text length, not measured by the agent.
- Only one prompt runs per provider at a time (serialized via prompt lock).

## Requirements

- Python 3.10 to 3.14
- Cursor CLI `agent` command available in `PATH`
- Cursor auth already set up (e.g. via `agent login`); this proxy does not perform ACP authentication

This project uses the official ACP Python SDK:

- [`agentclientprotocol/python-sdk`](https://github.com/agentclientprotocol/python-sdk)

## Install

```bash
uv sync --dev
```

Or with pip:

```bash
pip install -e .
```

## Run with TOML config

Use the sample config in `acp-to-api.toml`:

```toml
port = 11434
host = "0.0.0.0"
raw_acp = false
raw_rest = false

[providers.cursor]
command = "agent"
args = ["acp"]
```

Start:

```bash
uv run acp-to-api serve --config acp-to-api.toml
```

## Run with inline provider JSON

Pass providers directly from the CLI — repeat `--provider` for multiple:

```bash
uv run acp-to-api serve \
  --provider '{"name":"cursor","command":"agent","args":["acp"]}'
```

## Raw logging

- `--raw-acp` — logs ACP JSON-RPC traffic to/from the subprocess
- `--raw-rest` — logs incoming REST requests and outgoing responses

Both use Python's `logging` module at DEBUG level. Enable `DEBUG` to see output:

```bash
PYTHONLOGLEVEL=DEBUG uv run acp-to-api serve --config acp-to-api.toml --raw-acp --raw-rest
```

## OpenAI client example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/api/v1/cursor/openai",
    api_key="unused",
)

resp = client.chat.completions.create(
    model="cursor",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)

print(resp.choices[0].message.content)
```

## Run E2E tests

```bash
uv run pytest tests/test_e2e_cursor.py -q
```

## Publishing

```bash
uv build
uv publish
```

Or with standard tools:

```bash
python -m build
twine upload dist/*
```
