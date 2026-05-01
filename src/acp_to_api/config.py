from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, Field

from acp_to_api.dirs import state_dir as _default_state_dir

DEFAULT_STATE_DIR = _default_state_dir()


class ProviderConfig(BaseModel):
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None


class AppConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 11434
    raw_acp: bool = False
    raw_rest: bool = False
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    state_dir: Path = Field(default_factory=lambda: DEFAULT_STATE_DIR)
    config_path: Path | None = None


def _providers_from_toml(raw: dict[str, Any]) -> dict[str, ProviderConfig]:
    providers: dict[str, ProviderConfig] = {}
    providers_raw = raw.get("providers", {})
    if not isinstance(providers_raw, dict):
        return providers

    for name, cfg in providers_raw.items():
        if not isinstance(cfg, dict):
            continue
        providers[name] = ProviderConfig(name=name, **cfg)
    return providers


def load_config(path: Path | None) -> AppConfig:
    if path is None:
        return AppConfig()

    raw = tomllib.loads(path.read_text())
    cfg = AppConfig(
        host=raw.get("host", "127.0.0.1"),
        port=raw.get("port", 11434),
        raw_acp=raw.get("raw_acp", False),
        raw_rest=raw.get("raw_rest", False),
        providers=_providers_from_toml(raw),
        config_path=path.resolve(),
    )
    return cfg


def parse_provider_json(provider_json: str) -> ProviderConfig:
    raw = json.loads(provider_json)
    if "name" not in raw:
        raise ValueError("provider JSON must include 'name'")
    return ProviderConfig(**raw)


def merge_provider_configs(base: AppConfig, inline: list[ProviderConfig]) -> AppConfig:
    providers = dict(base.providers)
    for provider in inline:
        providers[provider.name] = provider
    return AppConfig(
        host=base.host,
        port=base.port,
        raw_acp=base.raw_acp,
        raw_rest=base.raw_rest,
        providers=providers,
        state_dir=base.state_dir,
        config_path=base.config_path,
    )


def validate_config(path: Path) -> AppConfig:
    """Load and validate a TOML config without side effects. Raises on invalid."""
    raw = tomllib.loads(path.read_text())
    providers = _providers_from_toml(raw)
    if not providers:
        raise ValueError(f"Config at {path} defines no providers")
    return AppConfig(
        host=raw.get("host", "127.0.0.1"),
        port=raw.get("port", 11434),
        raw_acp=raw.get("raw_acp", False),
        raw_rest=raw.get("raw_rest", False),
        providers=providers,
        config_path=path,
    )
