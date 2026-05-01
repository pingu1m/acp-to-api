# Using `acp-to-api` as backend for Codex and Claude Code

This document contains the **working** CLI configurations that were validated against this repo's local server.

## Prerequisites

1. Start the backend:

```bash
just serve
```

2. Confirm health:

```bash
curl -fsS http://127.0.0.1:11434/health
```

Expected:

```json
{"status":"ok"}
```

## Codex CLI config

Codex uses the OpenAI **Responses** wire API, so point it to:

- `http://127.0.0.1:11434/api/v1/cursor/openai`

Add this profile to `~/.codex/config.toml`:

```toml
[profiles.acp_local]
model = "cursor"
model_provider = "acp_local"
approval_policy = "never"

[model_providers.acp_local]
name = "ACP Local"
base_url = "http://127.0.0.1:11434/api/v1/cursor/openai"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
```

Run with:

```bash
printf 'Reply with exactly: codex-acp-ok\n' | \
  OPENAI_API_KEY=dummy \
  codex -p acp_local -a never -s read-only exec --skip-git-repo-check -C "$PWD" -
```

Validated result:

```text
codex-acp-ok
```

## Claude Code CLI config

Claude Code uses Anthropic Messages API shape, so point it to:

- `http://127.0.0.1:11434/api/v1/cursor/anthropic`

You can do this either with env vars or `--settings`.

### Option A: environment variables

```bash
printf 'Reply with exactly: claude-acp-ok\n' | \
  ANTHROPIC_API_KEY=dummy \
  ANTHROPIC_BASE_URL=http://127.0.0.1:11434/api/v1/cursor/anthropic \
  claude -p --model sonnet --permission-mode dontAsk
```

Validated result:

```text
claude-acp-ok
```

### Option B: settings file

Create a settings file (example: `/tmp/claude-acp-settings.json`):

```json
{
  "env": {
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:11434/api/v1/cursor/anthropic"
  }
}
```

Run with:

```bash
printf 'Reply with exactly: claude-settings-ok\n' | \
  claude -p --settings /tmp/claude-acp-settings.json --model sonnet --permission-mode dontAsk
```

Validated result:

```text
claude-settings-ok
```

## Notes

- Codex must use `wire_api = "responses"` (it does not support `chat_completions` wire mode).
- Claude Code must call the Anthropic-compatible messages route (`/v1/messages` under the configured base URL).
- The backend in this repo now exposes compatibility routes required by both CLIs:
  - `/api/v1/{provider}/openai/responses`
  - `/api/v1/{provider}/anthropic/v1/messages`
  - `/v1/messages` (default provider convenience route)
