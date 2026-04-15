from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
import uvicorn

from acp_to_api.config import AppConfig, load_config, merge_provider_configs, parse_provider_json
from acp_to_api.server import create_app

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _build_config(
    config_path: Optional[Path],
    providers_json: list[str],
    host: Optional[str],
    port: Optional[int],
    raw_acp: bool,
    raw_rest: bool,
) -> AppConfig:
    config = load_config(config_path)

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
    return config


@app.command()
def serve(
    config: Annotated[Optional[Path], typer.Option("--config", help="Path to TOML config file")] = None,
    provider: Annotated[
        list[str],
        typer.Option(
            "--provider",
            help='Provider JSON, e.g. {"name":"cursor","command":"agent","args":["acp"]}',
        ),
    ] = [],
    host: Annotated[Optional[str], typer.Option("--host", help="Bind host")] = None,
    port: Annotated[Optional[int], typer.Option("--port", help="Bind port")] = None,
    raw_acp: Annotated[bool, typer.Option("--raw-acp", help="Log raw ACP traffic")] = False,
    raw_rest: Annotated[bool, typer.Option("--raw-rest", help="Log raw REST requests/responses")] = False,
) -> None:
    """Run the ACP-to-OpenAI proxy server."""
    try:
        app_config = _build_config(config, provider, host, port, raw_acp, raw_rest)
    except (ValueError, json.JSONDecodeError, FileNotFoundError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if not app_config.providers:
        raise typer.BadParameter("No providers configured. Use --config or --provider.")

    fastapi_app = create_app(app_config)
    uvicorn.run(fastapi_app, host=app_config.host, port=app_config.port)


@app.command("providers")
def list_providers(
    config: Annotated[Optional[Path], typer.Option("--config", help="Path to TOML config file")] = None,
    provider: Annotated[list[str], typer.Option("--provider", help="Inline provider JSON")] = [],
) -> None:
    """List configured providers."""
    try:
        app_config = merge_provider_configs(
            load_config(config), [parse_provider_json(item) for item in provider]
        )
    except (ValueError, json.JSONDecodeError, FileNotFoundError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for name, cfg in app_config.providers.items():
        typer.echo(f"{name}: {cfg.command} {' '.join(cfg.args)}")


if __name__ == "__main__":
    app()
