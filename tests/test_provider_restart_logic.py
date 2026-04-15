from __future__ import annotations

import pytest

from acp_to_api.config import ProviderConfig
from acp_to_api.providers.cursor_acp import CursorACPProvider


class _FakeProc:
    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _make_provider() -> CursorACPProvider:
    return CursorACPProvider(ProviderConfig(name="cursor", command="agent", args=["acp"]))


@pytest.mark.asyncio
async def test_ensure_connection_restarts_when_not_alive() -> None:
    provider = _make_provider()
    provider._conn = None
    provider._proc = None

    calls: list[str] = []

    async def _fake_restart(reason: str) -> None:
        calls.append(reason)
        provider._conn = object()  # type: ignore[assignment]

    provider._restart_connection = _fake_restart  # type: ignore[method-assign]

    await provider._ensure_connection()

    assert len(calls) == 1
    assert "connection unavailable" in calls[0]


@pytest.mark.asyncio
async def test_ensure_connection_does_not_restart_when_alive() -> None:
    provider = _make_provider()
    provider._conn = object()  # type: ignore[assignment]
    provider._proc = _FakeProc(returncode=None)  # type: ignore[assignment]

    calls: list[str] = []

    async def _fake_restart(reason: str) -> None:
        calls.append(reason)

    provider._restart_connection = _fake_restart  # type: ignore[method-assign]

    await provider._ensure_connection()

    assert calls == []


@pytest.mark.asyncio
async def test_watch_subprocess_triggers_restart_on_exit() -> None:
    provider = _make_provider()
    proc = _FakeProc(returncode=7)
    provider._proc = proc  # type: ignore[assignment]
    provider._closed = False

    calls: list[str] = []

    async def _fake_restart(reason: str) -> None:
        calls.append(reason)

    provider._restart_connection = _fake_restart  # type: ignore[method-assign]

    await provider._watch_subprocess()

    assert len(calls) == 1
    assert "subprocess exited unexpectedly" in calls[0]
