from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _wait_for_health(url: str, timeout_s: float = 30.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"Server did not become healthy in {timeout_s}s")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server_port() -> int:
    return _free_port()


@pytest.fixture(scope="session")
def server_process(server_port: int) -> Iterator[subprocess.Popen[bytes]]:
    if shutil.which("agent") is None:
        pytest.skip("'agent' command not found; Cursor CLI is required for E2E tests")

    cmd = [
        sys.executable,
        "-m",
        "acp_to_api.cli",
        "serve",
        "--port",
        str(server_port),
        "--provider",
        '{"name":"cursor","command":"agent","args":["acp"]}',
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_for_health(f"http://127.0.0.1:{server_port}/health")
        yield proc
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
