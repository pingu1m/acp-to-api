# acp-to-api

[![PyPI version](https://img.shields.io/pypi/v/acp-to-api)](https://pypi.org/project/acp-to-api/)
[![CI](https://github.com/pingu1m/acp-to-api/actions/workflows/ci.yml/badge.svg)](https://github.com/pingu1m/acp-to-api/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Multi-provider ACP proxy CLI that exposes OpenAI-compatible REST APIs backed by Agent Client Protocol subprocess providers.

## Features

- **OpenAI-compatible API** -- `/chat/completions`, Responses API, and Anthropic `/v1/messages`
- **Multiple providers** -- Cursor, Kiro, Codex (via codex-acp), Claude Code (via claude-code-acp)
- **Runtime management** -- add, remove, and reload providers without restarting
- **Daemon mode** -- run as a background service with `start`/`stop`/`status`/`restart`
- **Auto-start** -- install as a macOS launchd or Linux systemd service
- **Web dashboard** -- embedded real-time trace viewer at `/dashboard`
- **Config hot-reload** -- watches config file for changes and applies them automatically
- **XDG directories** -- follows XDG Base Directory spec for config and state

## Quick start (no install needed)

Run directly with `uvx` -- nothing to install:

```bash
uvx acp-to-api serve --provider '{"name":"cursor","command":"agent","args":["acp"]}'
```

Or with a config file:

```bash
uvx acp-to-api serve --config acp-to-api.toml
```

## Install

```bash
uv add acp-to-api
```

Or with pip:

```bash
pip install acp-to-api
```

For development:

```bash
git clone https://github.com/pingu1m/acp-to-api.git
cd acp-to-api
uv sync --dev
```

## Requirements

- Python 3.10 to 3.14
- At least one ACP provider CLI installed (e.g. Cursor `agent`, `kiro-cli`, `codex-acp`, `claude-code-acp`)

This project uses the official [ACP Python SDK](https://github.com/agentclientprotocol/python-sdk).

## Configuration

Example `acp-to-api.toml`:

```toml
port = 11434
host = "127.0.0.1"

[providers.cursor]
command = "agent"
args = ["acp"]

[providers.kiro]
command = "kiro-cli"
args = ["acp", "--trust-all-tools"]
```

If no config file exists, `acp-to-api` creates a starter config at `~/.config/acp-to-api/config.toml` on first run.

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

## Daemon mode

```bash
uvx acp-to-api start --config acp-to-api.toml   # background
uvx acp-to-api status
uvx acp-to-api stop
uvx acp-to-api restart --config acp-to-api.toml
```

Install as a system service (auto-start on boot):

```bash
uvx acp-to-api setup --config acp-to-api.toml   # macOS launchd / Linux systemd
uvx acp-to-api setup --uninstall
```

## Dashboard

Open `http://127.0.0.1:11434/dashboard` to see live trace events, manage providers, and inspect request/response payloads.

## Raw logging

- `--raw-acp` -- logs ACP JSON-RPC traffic
- `--raw-rest` -- logs REST requests and responses

## Limitations

- **Tuning parameters** (`temperature`, `max_tokens`) are accepted for API compatibility but not forwarded to the ACP agent.
- **Token usage** in responses is estimated from text length, not measured by the agent.
- Only one prompt runs per provider at a time (serialized via prompt lock).

## Security

**acp-to-api is designed for local/trusted-network use only.** The API has no authentication; anyone who can reach the server can invoke providers, which spawn subprocesses.

- The default bind address is `127.0.0.1` (localhost only). Use `--host 0.0.0.0` only if you trust your network.
- If you need network access, place the server behind a reverse proxy with authentication.
- The dashboard and trace logs capture **full request/response bodies** (prompts, completions, tool calls). Be aware of this if you pipe sensitive data through the proxy.

## Run tests

```bash
uvx --from . --with pytest --with httpx --with pytest-asyncio pytest -v
```

## Lint and format

```bash
uvx ruff check .
uvx ruff format .
```

## Build and publish

```bash
uv build
uv publish
```

## License

[MIT](LICENSE)
