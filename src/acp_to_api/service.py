"""Generate and install launchd (macOS) / systemd (Linux) service definitions."""

from __future__ import annotations

import html
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import typer

from acp_to_api.dirs import state_dir as _default_state_dir

DEFAULT_STATE_DIR = _default_state_dir()

LABEL = "com.acp-to-api"
SYSTEMD_UNIT_NAME = "acp-to-api.service"


def detect_platform() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    raise RuntimeError(f"Unsupported platform: {sys.platform}")


def _resolve_exe() -> str:
    exe = shutil.which("acp-to-api")
    if exe:
        return exe
    return f"{sys.executable} -m acp_to_api.cli"


# -- macOS launchd ------------------------------------------------------------


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def generate_launchd_plist(
    config_path: Path,
    state_dir: Path,
    exe_path: str | None = None,
) -> str:
    exe = exe_path or _resolve_exe()
    log = str(state_dir / "acp-to-api.log")

    parts = exe.split() if " " in exe else [exe]
    all_args = [*parts, "serve", "--config", str(config_path), "--state-dir", str(state_dir)]
    program_args = "\n".join(f"        <string>{html.escape(p)}</string>" for p in all_args)

    safe_log = html.escape(log)

    return dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{html.escape(LABEL)}</string>
            <key>ProgramArguments</key>
            <array>
        {program_args}
            </array>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>StandardOutPath</key>
            <string>{safe_log}</string>
            <key>StandardErrorPath</key>
            <string>{safe_log}</string>
        </dict>
        </plist>
    """)


def install_launchd(config_path: Path, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    plist = _plist_path()
    plist.parent.mkdir(parents=True, exist_ok=True)

    content = generate_launchd_plist(config_path, state_dir)
    plist.write_text(content)
    typer.echo(f"Wrote {plist}")

    subprocess.run(["launchctl", "load", str(plist)], check=True)
    typer.echo(f"Loaded {LABEL} via launchctl")


def uninstall_launchd() -> None:
    plist = _plist_path()
    if plist.exists():
        subprocess.run(["launchctl", "unload", str(plist)], check=False)
        plist.unlink()
        typer.echo(f"Unloaded and removed {plist}")
    else:
        typer.echo(f"No plist found at {plist}")


# -- Linux systemd -------------------------------------------------------------


def _systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT_NAME


def generate_systemd_unit(
    config_path: Path,
    state_dir: Path,
    exe_path: str | None = None,
) -> str:
    exe = exe_path or _resolve_exe()
    log = str(state_dir / "acp-to-api.log")

    def _sd_quote(p: str) -> str:
        return f'"{p}"' if " " in p else p

    parts = exe.split() if " " in exe else [exe]
    all_args = [*parts, "serve", "--config", str(config_path), "--state-dir", str(state_dir)]
    exec_start = " ".join(_sd_quote(a) for a in all_args)

    return dedent(f"""\
        [Unit]
        Description=ACP-to-API Proxy
        After=network.target

        [Service]
        Type=simple
        ExecStart={exec_start}
        Restart=on-failure
        RestartSec=5
        StandardOutput=append:{log}
        StandardError=append:{log}

        [Install]
        WantedBy=default.target
    """)


def install_systemd(config_path: Path, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    unit = _systemd_unit_path()
    unit.parent.mkdir(parents=True, exist_ok=True)

    content = generate_systemd_unit(config_path, state_dir)
    unit.write_text(content)
    typer.echo(f"Wrote {unit}")

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT_NAME], check=True)
    typer.echo(f"Enabled and started {SYSTEMD_UNIT_NAME}")


def uninstall_systemd() -> None:
    unit = _systemd_unit_path()
    if unit.exists():
        subprocess.run(["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT_NAME], check=False)
        unit.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        typer.echo(f"Disabled, removed, and reloaded {unit}")
    else:
        typer.echo(f"No unit file found at {unit}")


# -- Unified entry point ------------------------------------------------------


def install_service(config_path: Path, state_dir: Path) -> None:
    platform = detect_platform()
    typer.echo(f"Detected platform: {platform}")
    if platform == "macos":
        install_launchd(config_path, state_dir)
    else:
        install_systemd(config_path, state_dir)
    typer.echo("Service installed. It will start automatically on login/boot.")


def uninstall_service() -> None:
    platform = detect_platform()
    typer.echo(f"Detected platform: {platform}")
    if platform == "macos":
        uninstall_launchd()
    else:
        uninstall_systemd()
    typer.echo("Service removed.")
