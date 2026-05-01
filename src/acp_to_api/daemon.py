"""Daemon utilities: background process spawning, PID file management, signal wiring.

Uses subprocess.Popen instead of os.fork() to avoid macOS Objective-C
runtime crashes (fork-after-ObjC-init is not safe on Darwin).
"""

from __future__ import annotations

import errno
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from acp_to_api.dirs import state_dir as _default_state_dir

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = _default_state_dir()
PID_FILENAME = "acp-to-api.pid"
LOG_FILENAME = "acp-to-api.log"


def ensure_state_dir(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)


def pid_file_path(state_dir: Path) -> Path:
    return state_dir / PID_FILENAME


def log_file_path(state_dir: Path) -> Path:
    return state_dir / LOG_FILENAME


def write_pid(state_dir: Path, pid: int | None = None) -> None:
    ensure_state_dir(state_dir)
    target = pid_file_path(state_dir)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(str(pid or os.getpid()))
    tmp.rename(target)


def read_pid(state_dir: Path) -> int | None:
    pf = pid_file_path(state_dir)
    if not pf.exists():
        return None
    try:
        return int(pf.read_text().strip())
    except (ValueError, OSError):
        return None


def remove_pid(state_dir: Path) -> None:
    pf = pid_file_path(state_dir)
    pf.unlink(missing_ok=True)


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            return False
        if err.errno == errno.EPERM:
            return True
        return False
    return True


def check_already_running(state_dir: Path) -> int | None:
    """Return PID of existing daemon if one is running, else None."""
    pid = read_pid(state_dir)
    if pid is not None and is_process_alive(pid):
        return pid
    if pid is not None:
        remove_pid(state_dir)
    return None


def spawn_daemon(
    serve_args: list[str],
    state_dir: Path,
) -> int:
    """Spawn ``acp-to-api serve`` as a detached background process.

    Returns the PID of the spawned process.
    """
    ensure_state_dir(state_dir)
    log_path = log_file_path(state_dir)

    cmd = [sys.executable, "-m", "acp_to_api.cli", "serve", *serve_args]

    log_fd = open(log_path, "a")  # noqa: SIM115

    proc = subprocess.Popen(
        cmd,
        stdout=log_fd,
        stderr=log_fd,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    log_fd.close()

    write_pid(state_dir, proc.pid)
    return proc.pid


def stop_daemon(state_dir: Path, timeout: float = 20.0) -> bool:
    """Send SIGTERM to running daemon, escalate to SIGKILL. Returns True if stopped."""
    pid = read_pid(state_dir)
    if pid is None:
        return False
    if not is_process_alive(pid):
        remove_pid(state_dir)
        return False

    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        os.kill(pid, signal.SIGTERM)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_process_alive(pid):
            remove_pid(state_dir)
            return True
        time.sleep(0.2)

    logger.warning("Daemon pid=%d did not exit after SIGTERM; sending SIGKILL", pid)
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(0.5)
    remove_pid(state_dir)
    return True


def install_signal_handlers(
    *,
    on_reload: object = None,
    on_shutdown: object = None,
) -> None:
    """Wire SIGHUP -> reload callback, SIGTERM/SIGINT -> shutdown callback."""
    if on_reload is not None:
        signal.signal(signal.SIGHUP, lambda *_: on_reload())  # type: ignore[operator]
    if on_shutdown is not None:

        def _shutdown_handler(*_: object) -> None:
            on_shutdown()  # type: ignore[operator]

        signal.signal(signal.SIGTERM, _shutdown_handler)
        signal.signal(signal.SIGINT, _shutdown_handler)
