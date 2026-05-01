# Changelog

## 0.1.1 (2026-05-01)

### Changed

- All justfile recipes and README examples now use `uvx` (zero-install runs)
- `serve` command allows starting with zero providers (add via API later)
- Added `AGENTS.md` and `.codex/config.toml` for Codex CLI integration

### Fixed

- CI test reliability: removed stub provider that caused 30s startup hang
- Foreground server fixture now captures stderr for faster CI debugging

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
