from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from acp_to_api.config import AppConfig, load_config, merge_provider_configs, parse_provider_json
from acp_to_api.daemon import (
    check_already_running,
    is_process_alive,
    log_file_path,
    pid_file_path,
    read_pid,
    remove_pid,
    spawn_daemon,
    stop_daemon,
)
from acp_to_api.dirs import (
    discover_config,
    ensure_default_config,
)
from acp_to_api.dirs import (
    state_dir as _default_state_dir,
)
from acp_to_api.server import create_app

DEFAULT_STATE_DIR = _default_state_dir()

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _resolve_config(
    explicit_config: Path | None,
    providers_json: list[str],
) -> Path | None:
    """Auto-discover a config file when none was explicitly provided.

    If no config exists anywhere and no inline providers were given,
    generates a starter config and tells the user to edit it.
    Returns None only when inline --provider args make a config file unnecessary.
    """
    if explicit_config is not None:
        return explicit_config

    if providers_json:
        return None

    found = discover_config()
    if found is not None:
        typer.echo(f"Using config: {found}")
        return found

    created = ensure_default_config()
    typer.echo(
        f"No config file found. Created a starter config at:\n\n"
        f"    {created}\n\n"
        f"Edit this file to add your providers, then run the command again."
    )
    raise typer.Exit(1)


def _build_config(
    config_path: Path | None,
    providers_json: list[str],
    host: str | None,
    port: int | None,
    raw_acp: bool,
    raw_rest: bool,
    state_dir: Path | None = None,
) -> AppConfig:
    resolved = _resolve_config(config_path, providers_json)
    config = load_config(resolved)

    inline_providers = [parse_provider_json(item) for item in providers_json]
    config = merge_provider_configs(config, inline_providers)

    if host is not None:
        config.host = host
    if port is not None:
        config.port = port
    if raw_acp:
        config.raw_acp = True
    if raw_rest:
        config.raw_rest = True
    if state_dir is not None:
        config.state_dir = state_dir
    return config


@app.command()
def serve(
    config: Annotated[Path | None, typer.Option("--config", help="Path to TOML config file")] = None,
    provider: Annotated[
        list[str],
        typer.Option(
            "--provider",
            help='Provider JSON, e.g. {"name":"cursor","command":"agent","args":["acp"]}',
        ),
    ] = [],
    host: Annotated[str | None, typer.Option("--host", help="Bind host")] = None,
    port: Annotated[int | None, typer.Option("--port", help="Bind port")] = None,
    raw_acp: Annotated[bool, typer.Option("--raw-acp", help="Log raw ACP traffic")] = False,
    raw_rest: Annotated[bool, typer.Option("--raw-rest", help="Log raw REST requests/responses")] = False,
    state_dir: Annotated[Path | None, typer.Option("--state-dir", help="State directory")] = None,
) -> None:
    """Run the ACP-to-OpenAI proxy server (foreground)."""
    try:
        app_config = _build_config(config, provider, host, port, raw_acp, raw_rest, state_dir)
    except (ValueError, json.JSONDecodeError, FileNotFoundError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if not app_config.providers:
        typer.echo("Warning: no providers configured. Add via --config, --provider, or the API.", err=True)

    fastapi_app = create_app(app_config)
    uvicorn.run(fastapi_app, host=app_config.host, port=app_config.port)


@app.command()
def start(
    config: Annotated[Path | None, typer.Option("--config", help="Path to TOML config file")] = None,
    provider: Annotated[
        list[str],
        typer.Option(
            "--provider",
            help='Provider JSON, e.g. {"name":"cursor","command":"agent","args":["acp"]}',
        ),
    ] = [],
    host: Annotated[str | None, typer.Option("--host", help="Bind host")] = None,
    port: Annotated[int | None, typer.Option("--port", help="Bind port")] = None,
    raw_acp: Annotated[bool, typer.Option("--raw-acp", help="Log raw ACP traffic")] = False,
    raw_rest: Annotated[bool, typer.Option("--raw-rest", help="Log raw REST requests/responses")] = False,
    state_dir: Annotated[Path | None, typer.Option("--state-dir", help="State directory")] = None,
) -> None:
    """Start the server as a background daemon."""
    resolved_state_dir = (state_dir or DEFAULT_STATE_DIR).resolve()

    existing_pid = check_already_running(resolved_state_dir)
    if existing_pid is not None:
        typer.echo(f"Daemon already running (pid={existing_pid})")
        raise typer.Exit(1)

    try:
        app_config = _build_config(config, provider, host, port, raw_acp, raw_rest, resolved_state_dir)
    except (ValueError, json.JSONDecodeError, FileNotFoundError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if not app_config.providers:
        raise typer.BadParameter("No providers configured. Use --config or --provider.")

    serve_args: list[str] = []
    if config is not None:
        serve_args.extend(["--config", str(config.resolve())])
    for p in provider:
        serve_args.extend(["--provider", p])
    if host is not None:
        serve_args.extend(["--host", host])
    if port is not None:
        serve_args.extend(["--port", str(port)])
    if raw_acp:
        serve_args.append("--raw-acp")
    if raw_rest:
        serve_args.append("--raw-rest")
    serve_args.extend(["--state-dir", str(resolved_state_dir)])

    pid = spawn_daemon(serve_args, resolved_state_dir)
    typer.echo(f"Daemon started (pid={pid}, state_dir={resolved_state_dir})")


@app.command()
def stop(
    state_dir: Annotated[Path | None, typer.Option("--state-dir", help="State directory")] = None,
) -> None:
    """Stop a running daemon."""
    resolved_state_dir = (state_dir or DEFAULT_STATE_DIR).resolve()

    pid = read_pid(resolved_state_dir)
    if pid is None:
        typer.echo("No daemon running (no PID file found)")
        raise typer.Exit(0)

    if not is_process_alive(pid):
        typer.echo(f"Stale PID file (pid={pid}); removing")
        remove_pid(resolved_state_dir)
        raise typer.Exit(0)

    typer.echo(f"Stopping daemon (pid={pid})...")
    stopped = stop_daemon(resolved_state_dir)
    if stopped:
        typer.echo("Daemon stopped")
    else:
        typer.echo("Failed to stop daemon")
        raise typer.Exit(1)


@app.command()
def status(
    state_dir: Annotated[Path | None, typer.Option("--state-dir", help="State directory")] = None,
) -> None:
    """Show daemon status."""
    resolved_state_dir = (state_dir or DEFAULT_STATE_DIR).resolve()

    pid = read_pid(resolved_state_dir)
    if pid is None:
        typer.echo("Status: not running (no PID file)")
        raise typer.Exit(1)

    if is_process_alive(pid):
        typer.echo(f"Status: running (pid={pid})")
        typer.echo(f"PID file: {pid_file_path(resolved_state_dir)}")
        typer.echo(f"Log file: {log_file_path(resolved_state_dir)}")
    else:
        typer.echo(f"Status: not running (stale PID file, pid={pid})")
        remove_pid(resolved_state_dir)
        raise typer.Exit(1)


@app.command()
def restart(
    config: Annotated[Path | None, typer.Option("--config", help="Path to TOML config file")] = None,
    provider: Annotated[
        list[str],
        typer.Option(
            "--provider",
            help='Provider JSON, e.g. {"name":"cursor","command":"agent","args":["acp"]}',
        ),
    ] = [],
    host: Annotated[str | None, typer.Option("--host", help="Bind host")] = None,
    port: Annotated[int | None, typer.Option("--port", help="Bind port")] = None,
    raw_acp: Annotated[bool, typer.Option("--raw-acp", help="Log raw ACP traffic")] = False,
    raw_rest: Annotated[bool, typer.Option("--raw-rest", help="Log raw REST requests/responses")] = False,
    state_dir: Annotated[Path | None, typer.Option("--state-dir", help="State directory")] = None,
) -> None:
    """Restart the daemon (stop + start)."""
    resolved_state_dir = (state_dir or DEFAULT_STATE_DIR).resolve()

    pid = read_pid(resolved_state_dir)
    if pid is not None and is_process_alive(pid):
        typer.echo(f"Stopping existing daemon (pid={pid})...")
        stop_daemon(resolved_state_dir)
        typer.echo("Stopped")

    try:
        app_config = _build_config(config, provider, host, port, raw_acp, raw_rest, resolved_state_dir)
    except (ValueError, json.JSONDecodeError, FileNotFoundError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if not app_config.providers:
        raise typer.BadParameter("No providers configured. Use --config or --provider.")

    serve_args: list[str] = []
    if config is not None:
        serve_args.extend(["--config", str(config.resolve())])
    for p in provider:
        serve_args.extend(["--provider", p])
    if host is not None:
        serve_args.extend(["--host", host])
    if port is not None:
        serve_args.extend(["--port", str(port)])
    if raw_acp:
        serve_args.append("--raw-acp")
    if raw_rest:
        serve_args.append("--raw-rest")
    serve_args.extend(["--state-dir", str(resolved_state_dir)])

    new_pid = spawn_daemon(serve_args, resolved_state_dir)
    typer.echo(f"Daemon started (pid={new_pid}, state_dir={resolved_state_dir})")


@app.command()
def setup(
    config: Annotated[Path | None, typer.Option("--config", help="Path to TOML config file")] = None,
    state_dir: Annotated[Path | None, typer.Option("--state-dir", help="State directory")] = None,
    uninstall: Annotated[bool, typer.Option("--uninstall", help="Remove the system service")] = False,
) -> None:
    """Install or remove the system service (launchd on macOS, systemd on Linux)."""
    from acp_to_api.service import install_service, uninstall_service

    if uninstall:
        uninstall_service()
        return

    if config is not None:
        resolved_config = config.resolve()
    else:
        found = discover_config()
        if found is not None:
            resolved_config = found.resolve()
            typer.echo(f"Using config: {resolved_config}")
        else:
            resolved_config = ensure_default_config().resolve()
            typer.echo(f"Created starter config at: {resolved_config}")

    if not resolved_config.exists():
        raise typer.BadParameter(f"Config file not found: {resolved_config}")

    resolved_state_dir = (state_dir or DEFAULT_STATE_DIR).resolve()
    install_service(resolved_config, resolved_state_dir)


@app.command("providers")
def list_providers(
    config: Annotated[Path | None, typer.Option("--config", help="Path to TOML config file")] = None,
    provider: Annotated[list[str], typer.Option("--provider", help="Inline provider JSON")] = [],
) -> None:
    """List configured providers."""
    try:
        app_config = merge_provider_configs(load_config(config), [parse_provider_json(item) for item in provider])
    except (ValueError, json.JSONDecodeError, FileNotFoundError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for name, cfg in app_config.providers.items():
        typer.echo(f"{name}: {cfg.command} {' '.join(cfg.args)}")


if __name__ == "__main__":
    app()
