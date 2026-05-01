from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from acp import PROTOCOL_VERSION, connect_to_agent, text_block
from acp.connection import StreamDirection, StreamEvent
from acp.core import ClientSideConnection
from acp.interfaces import Client
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    AvailableCommandsUpdate,
    ClientCapabilities,
    CurrentModeUpdate,
    DeniedOutcome,
    PermissionOption,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UserMessageChunk,
)

from acp_to_api.config import ProviderConfig
from acp_to_api.openai_models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    DeltaMessage,
    ToolCall,
    ToolCallFunction,
    ToolChoiceObject,
    Usage,
    message_content_to_text,
)
from acp_to_api.providers.base import BaseProvider

logger = logging.getLogger(__name__)

SessionUpdate = (
    UserMessageChunk
    | AgentMessageChunk
    | AgentThoughtChunk
    | ToolCallStart
    | ToolCallProgress
    | AgentPlanUpdate
    | AvailableCommandsUpdate
    | CurrentModeUpdate
)


@dataclass
class SessionAccumulator:
    queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    chunks: list[str] = field(default_factory=list)


class CursorACPProvider(BaseProvider):
    def __init__(self, config: ProviderConfig, raw_acp: bool = False, trace_hub: Any = None) -> None:
        self.name = config.name
        self._config = config
        self._raw_acp = raw_acp
        self._trace_hub = trace_hub
        self._prompt_lock = asyncio.Lock()
        self._states: dict[str, SessionAccumulator] = {}
        self._restart_lock = asyncio.Lock()
        self._monitor_task: asyncio.Task[None] | None = None
        self._closed = False
        self._available_models: list[dict[str, str]] = []

        self._proc: asyncio.subprocess.Process | None = None
        self._conn: ClientSideConnection | None = None
        self._client = _CursorACPClient(self)

    # -- Lifecycle -----------------------------------------------------------

    async def startup(self) -> None:
        self._closed = False
        await self._start_connection()

    async def shutdown(self) -> None:
        self._closed = True
        await self._close_connection()

    async def _start_connection(self) -> None:
        env = os.environ.copy()
        env.update(self._config.env)

        proc = await asyncio.create_subprocess_exec(
            self._config.command,
            *self._config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            cwd=self._config.cwd,
            env=env,
        )
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError(f"Provider {self.name}: subprocess missing stdin/stdout")

        observers = [self._stream_observer] if (self._raw_acp or self._trace_hub) else None
        conn = connect_to_agent(
            self._client,
            proc.stdin,
            proc.stdout,
            observers=observers,
        )

        await conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                fs={"readTextFile": False, "writeTextFile": False},
                terminal=False,
            ),
        )
        self._proc = proc
        self._conn = conn
        await self._discover_models()
        self._start_monitor()

    async def _close_connection(self) -> None:
        monitor_task = self._monitor_task
        self._monitor_task = None
        current_task = asyncio.current_task()
        if monitor_task is not None and monitor_task is not current_task and not monitor_task.done():
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task

        conn = self._conn
        self._conn = None
        if conn is not None:
            with contextlib.suppress(Exception):
                await conn.close()

        proc = self._proc
        self._proc = None
        if proc is not None and proc.returncode is None:
            proc.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5)
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    async def _discover_models(self) -> None:
        """Open a throwaway session to discover available models from the ACP agent."""
        conn = self._conn
        if conn is None:
            return
        try:
            cwd = self._config.cwd or os.getcwd()
            response = await conn.new_session(cwd=cwd, mcp_servers=[])
            self._update_models_from_session(response)
            # Close the probe session immediately
            with contextlib.suppress(Exception):
                await conn.close_session(session_id=response.session_id)
        except Exception as exc:
            logger.debug("Model discovery for provider '%s' failed: %s", self.name, exc)

    def _update_models_from_session(self, response: Any) -> None:
        """Extract available models from a NewSessionResponse."""
        models_state = getattr(response, "models", None)
        if models_state is None:
            return
        available = getattr(models_state, "available_models", None)
        if not available:
            return
        self._available_models = [
            {
                "id": m.model_id,
                "name": getattr(m, "name", m.model_id),
                "description": getattr(m, "description", None) or "",
            }
            for m in available
        ]
        logger.info(
            "Provider '%s' discovered %d models: %s",
            self.name,
            len(self._available_models),
            [m["id"] for m in self._available_models],
        )

    # -- Health monitoring ---------------------------------------------------

    def _start_monitor(self) -> None:
        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()
        self._monitor_task = asyncio.create_task(self._watch_subprocess())

    async def _watch_subprocess(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            await proc.wait()
        except asyncio.CancelledError:
            return

        if self._closed or proc is not self._proc:
            return
        logger.warning(
            "ACP subprocess for provider '%s' exited with code %s; restarting",
            self.name,
            proc.returncode,
        )
        await self._restart_connection(reason="subprocess exited unexpectedly")

    async def _restart_connection(self, reason: str) -> None:
        if self._closed:
            return
        async with self._restart_lock:
            if self._closed or self._is_connection_alive():
                return
            logger.info("Restarting ACP connection for provider '%s': %s", self.name, reason)
            await self._close_connection()
            await self._start_connection()

    def _is_connection_alive(self) -> bool:
        return self._conn is not None and self._proc is not None and self._proc.returncode is None

    async def _ensure_connection(self) -> ClientSideConnection:
        if not self._is_connection_alive():
            await self._restart_connection(reason="connection unavailable before request")
        if self._conn is None:
            raise RuntimeError(f"Provider {self.name} is not available")
        return self._conn

    # -- Session helper ------------------------------------------------------

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[tuple[str, SessionAccumulator, ClientSideConnection]]:
        conn = await self._ensure_connection()
        cwd = self._config.cwd or os.getcwd()
        response = await conn.new_session(cwd=cwd, mcp_servers=[])
        self._update_models_from_session(response)
        session_id = response.session_id
        state = SessionAccumulator()
        self._states[session_id] = state
        try:
            yield session_id, state, conn
        finally:
            self._states.pop(session_id, None)

    # -- Public API ----------------------------------------------------------

    def list_models(self) -> list[str]:
        if self._available_models:
            return [m["id"] for m in self._available_models]
        return [self.name]

    def list_models_detail(self) -> list[dict[str, str]]:
        """Return detailed model info (id, name, description) if available."""
        if self._available_models:
            return list(self._available_models)
        return [{"id": self.name, "name": self.name, "description": ""}]

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        prompt_text = _build_prompt_text(request)
        async with self._prompt_lock, self._session() as (session_id, state, conn):
            response = await conn.prompt(
                session_id=session_id,
                prompt=[text_block(prompt_text)],
            )

            raw_text = "".join(state.chunks).strip()
            message, finish_reason = _build_response_message(raw_text, request, response.stop_reason)

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=message,
                        finish_reason=finish_reason,
                    )
                ],
                usage=_estimate_usage(prompt_text, raw_text),
            )

    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        prompt_text = _build_prompt_text(request)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role="assistant"))],
        )

        async with self._prompt_lock, self._session() as (session_id, state, conn):
            prompt_task = asyncio.create_task(conn.prompt(session_id=session_id, prompt=[text_block(prompt_text)]))

            try:
                while True:
                    if prompt_task.done() and state.queue.empty():
                        break
                    try:
                        chunk = await asyncio.wait_for(state.queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue

                    yield ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=chunk))],
                    )

                response = await prompt_task
                yield ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=DeltaMessage(),
                            finish_reason=_map_stop_reason(response.stop_reason),
                        )
                    ],
                )
            except BaseException:
                prompt_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await prompt_task
                raise

    # -- ACP callbacks -------------------------------------------------------

    async def on_session_update(self, session_id: str, update: SessionUpdate) -> None:
        text = _extract_agent_message_text(update)
        if text is None:
            return
        state = self._states.get(session_id)
        if state is None:
            return
        state.chunks.append(text)
        await state.queue.put(text)

    async def _stream_observer(self, event: StreamEvent) -> None:
        direction = "->" if event.direction == StreamDirection.OUTGOING else "<-"
        body_json = json.dumps(event.message, ensure_ascii=True, default=str)

        if self._raw_acp:
            logger.debug("[ACP %s %s] %s", self.name, direction, body_json)

        if self._trace_hub is not None:
            from acp_to_api.dashboard import TraceEvent

            acp_method = ""
            if isinstance(event.message, dict):
                acp_method = event.message.get("method", "") or event.message.get("id", "")

            self._trace_hub.push(
                TraceEvent(
                    protocol="acp",
                    direction="outbound" if event.direction == StreamDirection.OUTGOING else "inbound",
                    provider=self.name,
                    method=str(acp_method),
                    summary=body_json[:200],
                    body=body_json,
                )
            )


class _CursorACPClient(Client):
    def __init__(self, provider: CursorACPProvider) -> None:
        self._provider = provider

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        for option in options:
            if option.kind in {"allow_once", "allow_always"} and isinstance(option.option_id, str):
                return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", optionId=option.option_id))
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def session_update(self, session_id: str, update: SessionUpdate, **kwargs: Any) -> None:
        await self._provider.on_session_update(session_id, update)

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "cursor/create_plan":
            return {"outcome": {"outcome": "accepted"}}
        if method == "cursor/ask_question":
            return {"outcome": {"outcome": "skipped", "reason": "non-interactive API mode"}}
        logger.debug("Unhandled ACP extension method: %s", method)
        return {"outcome": {"outcome": "cancelled"}}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        logger.debug("ACP extension notification: %s", method)


def _extract_agent_message_text(update: SessionUpdate) -> str | None:
    if not isinstance(update, AgentMessageChunk):
        return None
    content = update.content
    if isinstance(content, TextContentBlock):
        return content.text
    return None


def _build_prompt_text(request: ChatCompletionRequest) -> str:
    lines: list[str] = []
    for message in request.messages:
        text = message_content_to_text(message).strip()
        if text:
            lines.append(f"[{message.role}] {text}")

    if request.tools:
        target_tool = request.tools[0]
        if isinstance(request.tool_choice, ToolChoiceObject):
            for t in request.tools:
                if t.function.name == request.tool_choice.function.name:
                    target_tool = t
                    break

        if target_tool.function.parameters:
            schema_json = json.dumps(target_tool.function.parameters, indent=2)
            lines.append(
                "[INSTRUCTION] You MUST respond with ONLY a valid JSON object matching "
                "the following JSON schema. Do NOT include markdown, code blocks, or any "
                "extra text — output the raw JSON object only:\n" + schema_json
            )

    return "\n\n".join(lines)


def _build_response_message(
    raw_text: str,
    request: ChatCompletionRequest,
    stop_reason: str,
) -> tuple[ChatCompletionResponseMessage, str]:
    if not request.tools:
        return ChatCompletionResponseMessage(content=raw_text), _map_stop_reason(stop_reason)

    target_name = request.tools[0].function.name
    if isinstance(request.tool_choice, ToolChoiceObject):
        target_name = request.tool_choice.function.name

    json_str = _extract_json(raw_text)
    if json_str is not None:
        tool_call = ToolCall(
            id=f"call_{uuid.uuid4().hex[:24]}",
            function=ToolCallFunction(name=target_name, arguments=json_str),
        )
        return ChatCompletionResponseMessage(tool_calls=[tool_call]), "tool_calls"

    return ChatCompletionResponseMessage(content=raw_text), _map_stop_reason(stop_reason)


def _extract_json(text: str) -> str | None:
    """Extract the first valid JSON object from text that may include markdown or prose."""
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def _map_stop_reason(stop_reason: str) -> str:
    return {
        "end_turn": "stop",
        "max_tokens": "length",
        "cancelled": "stop",
        "refused": "stop",
    }.get(stop_reason, "stop")


def _estimate_usage(prompt_text: str, output_text: str) -> Usage:
    prompt_tokens = max(1, len(prompt_text) // 4) if prompt_text else 0
    completion_tokens = max(1, len(output_text) // 4) if output_text else 0
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
