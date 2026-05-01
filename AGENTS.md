# acp-to-api

Multi-provider ACP-to-OpenAI proxy CLI.

## Repo layout

```
src/acp_to_api/
├── cli.py              Typer CLI — serve, start, stop, status, restart, setup
├── server.py           FastAPI app, routes, middleware
├── config.py           AppConfig / ProviderConfig pydantic models
├── registry.py         ProviderRegistry — lifecycle, hot-reload, watcher
├── daemon.py           Background process spawn/stop, PID files
├── service.py          launchd / systemd service install/uninstall
├── dashboard.py        Embedded HTML dashboard + TraceHub
├── dirs.py             XDG paths, default config generation
├── openai_models.py    OpenAI-compatible pydantic models
└── providers/
    └── cursor_acp.py   ACP provider implementation
tests/                  E2E tests (daemon, OpenAI SDK, LangChain, etc.)
experiments/            Standalone uv-script experiments
```

## Build, test, lint

```bash
uv sync --dev          # install all deps
uv run ruff check .    # lint (fix with --fix)
uv run ruff format .   # auto-format
uv run pytest -v       # run all tests
uv run pytest tests/test_e2e_daemon.py -v   # daemon tests only
uv build               # build sdist + wheel
```

Or via justfile: `just install`, `just lint`, `just fmt`, `just test`, `just build`.

## Run the server

```bash
uv run acp-to-api serve --config acp-to-api.toml
uv run acp-to-api start --config acp-to-api.toml   # daemon mode
```

## Engineering conventions

- Python 3.10+. Use `X | Y` union syntax, not `typing.Union`.
- `uv` for all dependency management and execution — never raw pip.
- `ruff` for lint and format. Rules: E, F, W, I, UP. Line length 120.
- Use `typer.echo()` in CLI code, not `print()`.
- Default host is `127.0.0.1`. Never default to `0.0.0.0`.
- Atomic file writes: write to `.tmp`, then `rename()`.
- Dashboard HTML is fully self-contained in `dashboard.py` — no external assets.
- Single version source: `pyproject.toml` → `importlib.metadata` in `__init__.py`.
- Conventional commits: `fix:`, `feat:`, `docs:`, `refactor:`, `test:`.

## Constraints

- Do NOT add dependencies without discussing first.
- Do NOT bind to `0.0.0.0` by default.
- Do NOT use `innerHTML` for user-supplied data in dashboard — use `createElement`/`addEventListener`.
- Do NOT store secrets or API keys in source files.
- Keep `dashboard.py` E501-exempt (long HTML/CSS/JS lines are expected).

## What "done" means

- `uv run ruff check .` passes with zero errors.
- `uv run ruff format --check .` reports no changes needed.
- `uv run pytest tests/test_e2e_daemon.py -v` passes (the CI test suite).
- If touching CLI: verify `uv run acp-to-api --help` still works.
- If touching dashboard: verify `/dashboard` loads and WebSocket connects.
