"""ProviderRegistry: thread-safe mutable provider map with JSON cache and config watcher."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from acp_to_api.config import AppConfig, ProviderConfig, load_config

logger = logging.getLogger(__name__)

CACHE_FILENAME = "overrides.json"


class ProviderRegistry:
    """Central owner of all running providers with hot-reload and persistence."""

    def __init__(self, config: AppConfig, *, trace_hub: Any | None = None) -> None:
        self._lock = asyncio.Lock()
        self._config = config
        self._config_path: Path | None = config.config_path
        self._state_dir: Path = config.state_dir
        self._cache_path: Path = self._state_dir / CACHE_FILENAME
        self._raw_acp: bool = config.raw_acp
        self._trace_hub = trace_hub

        from acp_to_api.providers import BaseProvider

        self._providers: dict[str, BaseProvider] = {}
        self._provider_sources: dict[str, str] = {}
        self._watcher_task: asyncio.Task[None] | None = None

    # -- Public accessors -----------------------------------------------------

    def get(self, name: str) -> Any | None:
        return self._providers.get(name)

    def list_all(self) -> dict[str, Any]:
        return dict(self._providers)

    def list_info(self) -> list[dict[str, Any]]:
        result = []
        for name, provider in self._providers.items():
            result.append(
                {
                    "name": name,
                    "command": provider._config.command,
                    "args": provider._config.args,
                    "source": self._provider_sources.get(name, "unknown"),
                }
            )
        return result

    @property
    def default_provider_name(self) -> str | None:
        return next(iter(self._providers), None)

    # -- Lifecycle ------------------------------------------------------------

    async def startup(self) -> None:
        merged = self._merge_config_with_cache()
        for name, cfg in merged.items():
            source = self._provider_sources.get(name, "toml")
            try:
                await asyncio.wait_for(
                    self._start_provider(name, cfg, source=source),
                    timeout=30.0,
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.error("Skipping provider '%s' (failed to start): %s", name, exc)
        self._start_watcher()

    async def shutdown(self) -> None:
        self._stop_watcher()
        async with self._lock:
            for name in list(self._providers):
                await self._stop_provider(name)

    # -- Runtime mutations ----------------------------------------------------

    async def add_provider(self, name: str, cfg: ProviderConfig) -> None:
        async with self._lock:
            if name in self._providers:
                raise ValueError(f"Provider '{name}' already exists")
            try:
                await asyncio.wait_for(
                    self._start_provider(name, cfg, source="api"),
                    timeout=30.0,
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError(f"Provider '{name}' timed out during startup") from exc
            self._cache_add(name, cfg)

    async def remove_provider(self, name: str) -> None:
        async with self._lock:
            if name not in self._providers:
                raise KeyError(f"Provider '{name}' not found")
            await self._stop_provider(name)
            self._cache_remove(name)

    # -- Config reload --------------------------------------------------------

    async def reload_from_config(self) -> dict[str, Any]:
        """Re-read TOML, merge with cache, reconcile running providers. Returns diff."""
        if self._config_path is None:
            return {"error": "No config file path configured"}

        try:
            fresh_config = load_config(self._config_path)
        except Exception as exc:
            logger.warning("Config reload failed (keeping current config): %s", exc)
            return {"error": str(exc)}

        merged = self._merge_toml_providers_with_cache(
            dict(fresh_config.providers),
        )

        diff: dict[str, Any] = {"added": [], "removed": [], "changed": [], "unchanged": []}

        async with self._lock:
            current_names = set(self._providers)
            desired_names = set(merged)

            for name in desired_names - current_names:
                try:
                    await asyncio.wait_for(
                        self._start_provider(name, merged[name], source=self._provider_sources.get(name, "toml")),
                        timeout=30.0,
                    )
                    diff["added"].append(name)
                except (asyncio.TimeoutError, Exception) as exc:
                    logger.error("Reload: failed to start provider '%s': %s", name, exc)

            for name in current_names - desired_names:
                await self._stop_provider(name)
                diff["removed"].append(name)

            for name in current_names & desired_names:
                old_cfg = self._providers[name]._config
                new_cfg = merged[name]
                if old_cfg.command != new_cfg.command or old_cfg.args != new_cfg.args:
                    await self._stop_provider(name)
                    try:
                        await asyncio.wait_for(
                            self._start_provider(name, new_cfg, source=self._provider_sources.get(name, "toml")),
                            timeout=30.0,
                        )
                        diff["changed"].append(name)
                    except (asyncio.TimeoutError, Exception) as exc:
                        logger.error("Reload: failed to restart provider '%s': %s", name, exc)
                else:
                    diff["unchanged"].append(name)

        logger.info("Config reload diff: %s", diff)
        return diff

    # -- Internal provider lifecycle ------------------------------------------

    async def _start_provider(self, name: str, cfg: ProviderConfig, *, source: str = "toml") -> None:
        from acp_to_api.providers import CursorACPProvider

        provider = CursorACPProvider(cfg, raw_acp=self._raw_acp, trace_hub=self._trace_hub)
        try:
            await provider.startup()
        except Exception:
            logger.exception("Failed to start provider '%s'", name)
            raise
        self._providers[name] = provider
        self._provider_sources[name] = source
        logger.info("Provider '%s' started (source=%s)", name, source)

    async def _stop_provider(self, name: str) -> None:
        provider = self._providers.pop(name, None)
        self._provider_sources.pop(name, None)
        if provider is not None:
            try:
                await provider.shutdown()
            except Exception:
                logger.exception("Error shutting down provider '%s'", name)
            logger.info("Provider '%s' stopped", name)

    # -- Config + cache merging -----------------------------------------------

    def _merge_config_with_cache(self) -> dict[str, ProviderConfig]:
        toml_providers = dict(self._config.providers)
        return self._merge_toml_providers_with_cache(toml_providers)

    def _merge_toml_providers_with_cache(self, toml_providers: dict[str, ProviderConfig]) -> dict[str, ProviderConfig]:
        cache = self._load_cache()
        added = cache.get("added", {})
        removed = set(cache.get("removed", []))

        for name in toml_providers:
            self._provider_sources[name] = "toml"

        for name, raw_cfg in added.items():
            if name not in toml_providers:
                toml_providers[name] = ProviderConfig(name=name, **raw_cfg)
                self._provider_sources[name] = "api"

        for name in removed:
            toml_providers.pop(name, None)
            self._provider_sources.pop(name, None)

        return toml_providers

    # -- JSON cache I/O -------------------------------------------------------

    def _load_cache(self) -> dict[str, Any]:
        if not self._cache_path.exists():
            return {"added": {}, "removed": []}
        try:
            raw = json.loads(self._cache_path.read_text())
            if not isinstance(raw, dict):
                return {"added": {}, "removed": []}
            return raw
        except (json.JSONDecodeError, OSError):
            return {"added": {}, "removed": []}

    def _save_cache(self, cache: dict[str, Any]) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(cache, indent=2))

    def _cache_add(self, name: str, cfg: ProviderConfig) -> None:
        cache = self._load_cache()
        cache.setdefault("added", {})[name] = {
            "command": cfg.command,
            "args": cfg.args,
            "env": cfg.env,
        }
        removed = cache.get("removed", [])
        if name in removed:
            removed.remove(name)
        self._save_cache(cache)

    def _cache_remove(self, name: str) -> None:
        cache = self._load_cache()
        added = cache.get("added", {})
        if name in added:
            del added[name]
        else:
            removed = cache.setdefault("removed", [])
            if name not in removed:
                removed.append(name)
        self._save_cache(cache)

    # -- File watcher ---------------------------------------------------------

    def _start_watcher(self) -> None:
        if self._config_path is None:
            return
        if self._watcher_task is not None and not self._watcher_task.done():
            return
        self._watcher_task = asyncio.create_task(self._watch_config())

    def _stop_watcher(self) -> None:
        task = self._watcher_task
        self._watcher_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _watch_config(self) -> None:
        if self._config_path is None:
            return

        try:
            from watchfiles import awatch
        except ImportError:
            logger.warning("watchfiles not installed; config file watching disabled")
            return

        config_path_str = str(self._config_path.resolve())
        logger.info("Watching config file: %s", config_path_str)

        try:
            async for changes in awatch(self._config_path.parent, debounce=1000):
                relevant = any(str(Path(path).resolve()) == config_path_str for _change_type, path in changes)
                if not relevant:
                    continue
                logger.info("Config file change detected, reloading...")
                try:
                    await self.reload_from_config()
                except Exception:
                    logger.exception("Config reload after file change failed")
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Config watcher crashed")
