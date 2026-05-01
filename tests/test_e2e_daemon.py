"""End-to-end tests for daemon lifecycle, provider management API, config
hot-reload, and JSON cache persistence.

All tests use isolated temp directories so they never touch ~/.acp-to-api or
the project config.  The server is started with the real ``agent`` ACP CLI
when available, otherwise the provider management / daemon tests still run
with a stub provider that will fail to start (we skip model-level assertions
in that case).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

HAS_AGENT = shutil.which("agent") is not None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(port: int, timeout: float = 15.0) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"Server on port {port} did not become healthy in {timeout}s")


def _wait_for_down(port: int, timeout: float = 10.0) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(url, timeout=1.0)
        except Exception:
            return
        time.sleep(0.3)
    raise RuntimeError(f"Server on port {port} still up after {timeout}s")


def _run_cli(*args: str, timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "acp_to_api.cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
    )


def _write_toml(path: Path, providers: dict[str, dict]) -> None:
    lines = ["port = 0", 'host = "127.0.0.1"', ""]
    for name, cfg in providers.items():
        lines.append(f"[providers.{name}]")
        lines.append(f'command = "{cfg["command"]}"')
        args_str = ", ".join(f'"{a}"' for a in cfg.get("args", []))
        lines.append(f"args = [{args_str}]")
        lines.append("")
    path.write_text("\n".join(lines))


@dataclass
class DaemonEnv:
    state_dir: Path
    config_path: Path
    port: int

    def cli(self, *args: str, timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
        return _run_cli(*args, "--state-dir", str(self.state_dir), timeout=timeout)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture
def daemon_env(tmp_path: Path) -> Iterator[DaemonEnv]:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    config_path = tmp_path / "test.toml"
    port = _find_free_port()

    providers: dict[str, dict] = {}
    if HAS_AGENT:
        providers["cursor"] = {"command": "agent", "args": ["acp"]}
    else:
        providers["stub"] = {"command": "echo", "args": ["stub"]}

    _write_toml(config_path, providers)
    config_path.write_text(config_path.read_text().replace("port = 0", f"port = {port}"))

    env = DaemonEnv(state_dir=state_dir, config_path=config_path, port=port)
    yield env

    pid_file = state_dir / "acp-to-api.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGKILL)
        except (ValueError, OSError):
            pass
        pid_file.unlink(missing_ok=True)


# -- Foreground serve fixture (for backward-compat + API tests) ---------------


@pytest.fixture
def foreground_server(daemon_env: DaemonEnv) -> Iterator[DaemonEnv]:
    """Start the server in foreground mode as a subprocess."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "acp_to_api.cli",
            "serve",
            "--config",
            str(daemon_env.config_path),
            "--state-dir",
            str(daemon_env.state_dir),
        ],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(daemon_env.port)
        yield daemon_env
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


# =============================================================================
# Group 1: Daemon lifecycle (CLI)
# =============================================================================


class TestDaemonLifecycle:
    @pytest.mark.skipif(not HAS_AGENT, reason="agent CLI required")
    def test_start_creates_pid_and_serves(self, daemon_env: DaemonEnv) -> None:
        result = daemon_env.cli(
            "start",
            "--config",
            str(daemon_env.config_path),
            "--port",
            str(daemon_env.port),
        )
        assert result.returncode == 0, result.stderr

        time.sleep(1)
        _wait_for_health(daemon_env.port)

        pid_file = daemon_env.state_dir / "acp-to-api.pid"
        assert pid_file.exists()
        pid = int(pid_file.read_text().strip())
        assert pid > 0

        r = httpx.get(f"{daemon_env.base_url}/health", timeout=5.0)
        assert r.status_code == 200

    @pytest.mark.skipif(not HAS_AGENT, reason="agent CLI required")
    def test_status_reports_running(self, daemon_env: DaemonEnv) -> None:
        daemon_env.cli(
            "start",
            "--config",
            str(daemon_env.config_path),
            "--port",
            str(daemon_env.port),
        )
        time.sleep(1)
        _wait_for_health(daemon_env.port)

        result = daemon_env.cli("status")
        assert result.returncode == 0
        assert "running" in result.stdout.lower()

    @pytest.mark.skipif(not HAS_AGENT, reason="agent CLI required")
    def test_stop_kills_and_cleans_pid(self, daemon_env: DaemonEnv) -> None:
        daemon_env.cli(
            "start",
            "--config",
            str(daemon_env.config_path),
            "--port",
            str(daemon_env.port),
        )
        time.sleep(1)
        _wait_for_health(daemon_env.port)

        result = daemon_env.cli("stop")
        assert result.returncode == 0
        assert "stopped" in result.stdout.lower()

        pid_file = daemon_env.state_dir / "acp-to-api.pid"
        assert not pid_file.exists()

        _wait_for_down(daemon_env.port)

    @pytest.mark.skipif(not HAS_AGENT, reason="agent CLI required")
    def test_restart_swaps_process(self, daemon_env: DaemonEnv) -> None:
        daemon_env.cli(
            "start",
            "--config",
            str(daemon_env.config_path),
            "--port",
            str(daemon_env.port),
        )
        time.sleep(1)
        _wait_for_health(daemon_env.port)

        pid_file = daemon_env.state_dir / "acp-to-api.pid"
        old_pid = int(pid_file.read_text().strip())

        daemon_env.cli(
            "restart",
            "--config",
            str(daemon_env.config_path),
            "--port",
            str(daemon_env.port),
        )
        time.sleep(1)
        _wait_for_health(daemon_env.port)

        new_pid = int(pid_file.read_text().strip())
        assert new_pid != old_pid

    def test_stop_when_not_running(self, daemon_env: DaemonEnv) -> None:
        result = daemon_env.cli("stop")
        assert result.returncode == 0
        assert "no daemon running" in result.stdout.lower() or "no pid" in result.stdout.lower()

    @pytest.mark.skipif(not HAS_AGENT, reason="agent CLI required")
    def test_start_when_already_running(self, daemon_env: DaemonEnv) -> None:
        daemon_env.cli(
            "start",
            "--config",
            str(daemon_env.config_path),
            "--port",
            str(daemon_env.port),
        )
        time.sleep(1)
        _wait_for_health(daemon_env.port)

        result = daemon_env.cli(
            "start",
            "--config",
            str(daemon_env.config_path),
            "--port",
            str(daemon_env.port),
        )
        assert result.returncode != 0
        assert "already running" in result.stdout.lower()


# =============================================================================
# Group 2: Provider management API
# =============================================================================


class TestProviderManagementAPI:
    def test_list_providers_returns_initial(self, foreground_server: DaemonEnv) -> None:
        r = httpx.get(f"{foreground_server.base_url}/api/v1/providers", timeout=5.0)
        assert r.status_code == 200
        providers = r.json()
        assert isinstance(providers, list)
        assert len(providers) >= 1
        names = [p["name"] for p in providers]
        if HAS_AGENT:
            assert "cursor" in names
        assert all(p["source"] in ("toml", "api") for p in providers)

    def test_add_provider_via_api(self, foreground_server: DaemonEnv) -> None:
        payload = {"name": "echo_test", "command": "echo", "args": ["hello"]}
        r = httpx.post(
            f"{foreground_server.base_url}/api/v1/providers",
            json=payload,
            timeout=35.0,
        )
        # Non-ACP commands will fail to start (422); ACP commands succeed (201)
        assert r.status_code in (201, 422), r.text
        if r.status_code == 201:
            r2 = httpx.get(f"{foreground_server.base_url}/api/v1/providers", timeout=5.0)
            names = [p["name"] for p in r2.json()]
            assert "echo_test" in names

    @pytest.mark.skipif(not HAS_AGENT, reason="agent CLI required for provider lifecycle")
    def test_add_and_remove_provider_via_api(self, foreground_server: DaemonEnv) -> None:
        r = httpx.post(
            f"{foreground_server.base_url}/api/v1/providers",
            json={"name": "cursor2", "command": "agent", "args": ["acp"]},
            timeout=35.0,
        )
        assert r.status_code == 201, r.text

        r2 = httpx.get(f"{foreground_server.base_url}/api/v1/providers", timeout=5.0)
        names = [p["name"] for p in r2.json()]
        assert "cursor2" in names

        r3 = httpx.delete(
            f"{foreground_server.base_url}/api/v1/providers/cursor2",
            timeout=5.0,
        )
        assert r3.status_code == 200

        r4 = httpx.get(f"{foreground_server.base_url}/api/v1/providers", timeout=5.0)
        names = [p["name"] for p in r4.json()]
        assert "cursor2" not in names

    def test_remove_nonexistent_provider_returns_404(self, foreground_server: DaemonEnv) -> None:
        r = httpx.delete(
            f"{foreground_server.base_url}/api/v1/providers/nope",
            timeout=5.0,
        )
        assert r.status_code == 404

    @pytest.mark.skipif(not HAS_AGENT, reason="agent CLI required for duplicate check")
    def test_add_duplicate_provider_returns_409(self, foreground_server: DaemonEnv) -> None:
        r = httpx.post(
            f"{foreground_server.base_url}/api/v1/providers",
            json={"name": "cursor", "command": "echo", "args": []},
            timeout=35.0,
        )
        assert r.status_code == 409


# =============================================================================
# Group 3: Config file hot-reload
# =============================================================================


class TestConfigHotReload:
    @pytest.mark.skipif(not HAS_AGENT, reason="agent CLI required for provider reload")
    def test_reload_api_adds_provider(self, foreground_server: DaemonEnv) -> None:
        toml_text = foreground_server.config_path.read_text()
        toml_text += '\n[providers.newone]\ncommand = "agent"\nargs = ["acp"]\n'
        foreground_server.config_path.write_text(toml_text)

        r = httpx.post(
            f"{foreground_server.base_url}/api/v1/providers/reload",
            timeout=60.0,
        )
        assert r.status_code == 200
        diff = r.json()
        assert "newone" in diff.get("added", [])

        r2 = httpx.get(f"{foreground_server.base_url}/api/v1/providers", timeout=5.0)
        names = [p["name"] for p in r2.json()]
        assert "newone" in names

    @pytest.mark.skipif(not HAS_AGENT, reason="agent CLI required for provider lifecycle")
    def test_config_change_removes_provider_on_reload(self, foreground_server: DaemonEnv) -> None:
        original_text = foreground_server.config_path.read_text()
        with_extra = original_text + '\n[providers.tempone]\ncommand = "agent"\nargs = ["acp"]\n'
        foreground_server.config_path.write_text(with_extra)
        httpx.post(f"{foreground_server.base_url}/api/v1/providers/reload", timeout=60.0)

        r = httpx.get(f"{foreground_server.base_url}/api/v1/providers", timeout=5.0)
        assert "tempone" in [p["name"] for p in r.json()]

        foreground_server.config_path.write_text(original_text)

        r2 = httpx.post(f"{foreground_server.base_url}/api/v1/providers/reload", timeout=60.0)
        assert r2.status_code == 200
        diff = r2.json()
        assert "tempone" in diff.get("removed", [])

    def test_invalid_config_keeps_running(self, foreground_server: DaemonEnv) -> None:
        foreground_server.config_path.write_text("this is not valid toml [[[")

        r = httpx.post(
            f"{foreground_server.base_url}/api/v1/providers/reload",
            timeout=10.0,
        )
        assert r.status_code == 200
        assert "error" in r.json()

        r2 = httpx.get(f"{foreground_server.base_url}/health", timeout=5.0)
        assert r2.status_code == 200


# =============================================================================
# Group 4: JSON cache persistence
# =============================================================================


class TestJSONCachePersistence:
    @pytest.mark.skipif(not HAS_AGENT, reason="agent CLI required for provider add")
    def test_cache_file_written_on_add(self, foreground_server: DaemonEnv) -> None:
        r = httpx.post(
            f"{foreground_server.base_url}/api/v1/providers",
            json={"name": "cached_prov", "command": "agent", "args": ["acp"]},
            timeout=35.0,
        )
        assert r.status_code == 201

        cache_path = foreground_server.state_dir / "overrides.json"
        assert cache_path.exists()
        cache = json.loads(cache_path.read_text())
        assert "cached_prov" in cache.get("added", {})
        assert cache["added"]["cached_prov"]["command"] == "agent"

    @pytest.mark.skipif(not HAS_AGENT, reason="agent CLI required for provider lifecycle")
    def test_cache_file_written_on_remove(self, foreground_server: DaemonEnv) -> None:
        httpx.delete(
            f"{foreground_server.base_url}/api/v1/providers/cursor",
            timeout=5.0,
        )

        cache_path = foreground_server.state_dir / "overrides.json"
        assert cache_path.exists()
        cache = json.loads(cache_path.read_text())
        assert "cursor" in cache.get("removed", [])


# =============================================================================
# Group 5: Backward compatibility
# =============================================================================


class TestBackwardCompatibility:
    def test_serve_foreground_still_works(self, foreground_server: DaemonEnv) -> None:
        r = httpx.get(f"{foreground_server.base_url}/health", timeout=5.0)
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_health_endpoint(self, foreground_server: DaemonEnv) -> None:
        r = httpx.get(f"{foreground_server.base_url}/health", timeout=5.0)
        assert r.status_code == 200

    @pytest.mark.skipif(not HAS_AGENT, reason="agent CLI required for model listing")
    def test_models_endpoint_via_registry(self, foreground_server: DaemonEnv) -> None:
        r = httpx.get(
            f"{foreground_server.base_url}/api/v1/cursor/openai/models",
            timeout=10.0,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data.get("data", [])) > 0
