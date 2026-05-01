"""XDG Base Directory paths and default config management."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "acp-to-api"
CWD_CONFIG_NAME = "acp-to-api.toml"
XDG_CONFIG_NAME = "config.toml"

STARTER_CONFIG = """\
# acp-to-api configuration
# https://github.com/pingu1m/acp-to-api
#
# Each provider runs as an ACP (Agent Client Protocol) subprocess.
# Uncomment the providers you have installed.

port = 11434
host = "127.0.0.1"

# --- Native ACP providers (built-in "acp" subcommand) ---

# [providers.cursor]
# command = "agent"
# args = ["acp"]

# [providers.kiro]
# command = "kiro-cli"
# args = ["acp", "--trust-all-tools"]

# --- ACP adapters by Zed Industries (install separately) ---
# codex-acp:        npm i -g @zed-industries/codex-acp
# claude-code-acp:  npm i -g @agentclientprotocol/claude-agent-acp

# [providers.codex]
# command = "codex-acp"
# args = []

# [providers.claude-code]
# command = "claude-code-acp"
# args = []
"""


def config_dir() -> Path:
    """$XDG_CONFIG_HOME/acp-to-api (default ~/.config/acp-to-api)."""
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP_NAME


def state_dir() -> Path:
    """$XDG_STATE_HOME/acp-to-api (default ~/.local/state/acp-to-api)."""
    base = os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state")
    return Path(base) / APP_NAME


def default_config_path() -> Path:
    return config_dir() / XDG_CONFIG_NAME


def discover_config() -> Path | None:
    """Search for an existing config file: CWD first, then XDG."""
    cwd_path = Path.cwd() / CWD_CONFIG_NAME
    if cwd_path.is_file():
        return cwd_path

    xdg_path = default_config_path()
    if xdg_path.is_file():
        return xdg_path

    return None


def ensure_default_config() -> Path:
    """Create a starter config at the XDG path if it doesn't exist. Returns the path."""
    p = default_config_path()
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(STARTER_CONFIG)
    return p
