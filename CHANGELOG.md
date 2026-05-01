# Changelog

## 0.1.0 (2026-05-01)

Initial release.

### Features

- Multi-provider ACP proxy exposing OpenAI-compatible `/chat/completions`, Responses (`/openai/responses`), and Anthropic (`/v1/messages`) APIs
- Provider management via REST API (add, remove, reload at runtime)
- Config hot-reload via file watcher (watchfiles)
- Daemon mode with `start`, `stop`, `status`, `restart` CLI commands
- macOS (launchd) and Linux (systemd) auto-start service installation
- Embedded web dashboard at `/dashboard` with real-time trace streaming over WebSocket
- XDG Base Directory support for config and state files
- Auto-generated starter config on first run
- Support for Cursor, Kiro, Codex (via codex-acp), and Claude Code (via claude-code-acp) providers
