from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from acp_to_api.config import AppConfig, ProviderConfig
from acp_to_api.dashboard import DASHBOARD_HTML, TraceEvent, TraceHub
from acp_to_api.openai_models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    OpenAIModel,
    OpenAIModelsResponse,
)
from acp_to_api.providers import BaseProvider
from acp_to_api.registry import ProviderRegistry

logger = logging.getLogger(__name__)


def create_app(config: AppConfig) -> FastAPI:
    trace_hub = TraceHub()
    registry = ProviderRegistry(config, trace_hub=trace_hub)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await registry.startup()
        try:
            yield
        finally:
            await registry.shutdown()

    from acp_to_api import __version__

    app = FastAPI(title="acp-to-api", version=__version__, lifespan=lifespan)
    app.state.registry = registry
    app.state.trace_hub = trace_hub
    app.state.raw_rest = config.raw_rest
    app.state.config = config
    app.state.start_time = time.time()

    @app.middleware("http")
    async def _trace_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.url.path in ("/dashboard", "/api/v1/ws/traces", "/health"):
            return await call_next(request)

        t0 = time.time()
        body = await request.body()
        body_text = body.decode("utf-8", errors="replace") if body else ""

        if config.raw_rest:
            logger.debug("[REST ->] %s %s %s", request.method, request.url.path, body_text)

        trace_hub.push(
            TraceEvent(
                protocol="rest",
                direction="inbound",
                provider=_provider_from_path(request.url.path),
                method=f"{request.method} {request.url.path}",
                summary=body_text[:200],
                body=body_text,
            )
        )

        response = await call_next(request)

        if isinstance(response, StreamingResponse):
            return response

        content = b""
        async for chunk_bytes in response.body_iterator:
            if isinstance(chunk_bytes, str):
                content += chunk_bytes.encode("utf-8")
            else:
                content += chunk_bytes
        resp_text = content.decode("utf-8", errors="replace")
        duration = (time.time() - t0) * 1000

        if config.raw_rest:
            logger.debug("[REST <-] %s %s", response.status_code, resp_text)

        trace_hub.push(
            TraceEvent(
                protocol="rest",
                direction="outbound",
                provider=_provider_from_path(request.url.path),
                method=f"{request.method} {request.url.path}",
                summary=resp_text[:200],
                body=resp_text,
                status=response.status_code,
                duration_ms=round(duration, 1),
            )
        )

        headers = dict(response.headers)
        return Response(
            content=content,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )

    @app.get("/dashboard")
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/api/v1/config")
    async def get_config() -> JSONResponse:
        return JSONResponse(
            content={
                "host": config.host,
                "port": config.port,
                "raw_acp": config.raw_acp,
                "raw_rest": config.raw_rest,
                "config_path": str(config.config_path) if config.config_path else None,
                "state_dir": str(config.state_dir),
            }
        )

    @app.websocket("/api/v1/ws/traces")
    async def ws_traces(websocket: WebSocket) -> None:
        await websocket.accept()
        q = trace_hub.subscribe()
        if q is None:
            await websocket.close(code=1013, reason="Too many connections")
            return
        try:
            await websocket.send_text(json.dumps(trace_hub.recent(200)))
            while True:
                event = await q.get()
                await websocket.send_text(json.dumps(event.to_dict()))
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            trace_hub.unsubscribe(q)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/{provider_name}/openai/models")
    async def list_models(provider_name: str) -> OpenAIModelsResponse:
        provider = _get_provider(app, provider_name)
        models = [
            OpenAIModel(id=model_name, created=0, owned_by=f"acp-to-api:{provider_name}")
            for model_name in provider.list_models()
        ]
        return OpenAIModelsResponse(data=models)

    @app.get("/api/v1/{provider_name}/openai/models/{model_id:path}")
    async def retrieve_model(provider_name: str, model_id: str) -> OpenAIModel:
        provider = _get_provider(app, provider_name)
        available = provider.list_models()
        if model_id not in available:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        return OpenAIModel(id=model_id, created=0, owned_by=f"acp-to-api:{provider_name}")

    @app.post("/api/v1/{provider_name}/openai/chat/completions")
    async def chat_completions(provider_name: str, request: ChatCompletionRequest) -> Response:
        provider = _get_provider(app, provider_name)
        if request.stream:
            return StreamingResponse(
                _stream_sse(provider.chat_completion_stream(request), app.state.raw_rest),
                media_type="text/event-stream",
            )
        response = await provider.chat_completion(request)
        return JSONResponse(content=response.model_dump(exclude_none=True))

    @app.post("/api/v1/{provider_name}/openai/responses")
    async def responses_api(provider_name: str, request: Request) -> Response:
        provider = _get_provider(app, provider_name)
        payload = await _read_json_payload(request)
        chat_request = _responses_payload_to_chat_request(payload, provider_name)

        if chat_request.stream:
            return StreamingResponse(
                _stream_responses_sse(
                    provider.chat_completion_stream(chat_request),
                    model=chat_request.model,
                    raw_rest=app.state.raw_rest,
                ),
                media_type="text/event-stream",
            )

        chat_response = await provider.chat_completion(chat_request)
        return JSONResponse(content=_responses_json_from_chat_response(chat_response))

    @app.post("/v1/messages")
    async def anthropic_messages_root(request: Request) -> Response:
        provider = _get_default_provider(app)
        payload = await _read_json_payload(request)
        return await _anthropic_messages_response(
            provider=provider,
            payload=payload,
            provider_name=provider.name,
            raw_rest=app.state.raw_rest,
        )

    @app.post("/api/v1/{provider_name}/anthropic/v1/messages")
    async def anthropic_messages_provider(provider_name: str, request: Request) -> Response:
        provider = _get_provider(app, provider_name)
        payload = await _read_json_payload(request)
        return await _anthropic_messages_response(
            provider=provider,
            payload=payload,
            provider_name=provider_name,
            raw_rest=app.state.raw_rest,
        )

    # -- Provider management endpoints ----------------------------------------

    @app.get("/api/v1/providers")
    async def list_providers_api() -> JSONResponse:
        return JSONResponse(content=registry.list_info())

    @app.post("/api/v1/providers")
    async def add_provider_api(request: Request) -> JSONResponse:
        payload = await _read_json_payload(request)
        name = payload.get("name")
        command = payload.get("command")
        if not name or not command:
            raise HTTPException(status_code=400, detail="'name' and 'command' are required")
        cfg = ProviderConfig(
            name=name,
            command=command,
            args=payload.get("args", []),
            env=payload.get("env", {}),
            cwd=payload.get("cwd"),
        )
        try:
            await registry.add_provider(name, cfg)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(content={"status": "added", "name": name}, status_code=201)

    @app.delete("/api/v1/providers/{name}")
    async def remove_provider_api(name: str) -> JSONResponse:
        try:
            await registry.remove_provider(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(content={"status": "removed", "name": name})

    @app.post("/api/v1/providers/reload")
    async def reload_providers_api() -> JSONResponse:
        diff = await registry.reload_from_config()
        return JSONResponse(content=diff)

    return app


def _get_provider(app: FastAPI, provider_name: str) -> BaseProvider:
    provider = app.state.registry.get(provider_name)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_name}")
    return provider


def _get_default_provider(app: FastAPI) -> BaseProvider:
    provider_name = app.state.registry.default_provider_name
    if not provider_name:
        raise HTTPException(status_code=503, detail="No providers configured")
    return _get_provider(app, provider_name)


async def _read_json_payload(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return payload


def _normalize_role(role: Any) -> str:
    if role in {"system", "developer", "user", "assistant", "tool"}:
        return str(role)
    return "user"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text_value = item.get("text")
            if isinstance(text_value, str):
                parts.append(text_value)
                continue
            maybe_input_text = item.get("input_text")
            if isinstance(maybe_input_text, str):
                parts.append(maybe_input_text)
                continue
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def _responses_payload_to_chat_request(payload: dict[str, Any], provider_name: str) -> ChatCompletionRequest:
    model = str(payload.get("model") or provider_name)
    stream = bool(payload.get("stream", True))
    max_tokens_raw = payload.get("max_output_tokens")
    if max_tokens_raw is None:
        max_tokens_raw = payload.get("max_tokens")
    max_tokens: int | None = max_tokens_raw if isinstance(max_tokens_raw, int) else None

    messages: list[ChatMessage] = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append(ChatMessage(role="system", content=instructions))

    input_payload = payload.get("input")
    if isinstance(input_payload, str):
        text = input_payload.strip()
        if text:
            messages.append(ChatMessage(role="user", content=text))
    elif isinstance(input_payload, list):
        for item in input_payload:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message" or "role" in item:
                role = _normalize_role(item.get("role", "user"))
                content_text = _content_to_text(item.get("content"))
                if content_text:
                    messages.append(ChatMessage(role=role, content=content_text))

    if not messages:
        raise HTTPException(
            status_code=400,
            detail="Responses request must include at least one message in `input` or `instructions`.",
        )

    return ChatCompletionRequest(
        model=model,
        messages=messages,
        stream=stream,
        max_completion_tokens=max_tokens,
    )


def _responses_json_from_chat_response(chat_response: ChatCompletionResponse) -> dict[str, Any]:
    text = chat_response.choices[0].message.content or ""
    resp_id = f"resp_{uuid.uuid4().hex}"
    msg_id = f"msg_{uuid.uuid4().hex}"
    return {
        "id": resp_id,
        "object": "response",
        "created_at": chat_response.created,
        "status": "completed",
        "model": chat_response.model,
        "output": [
            {
                "id": msg_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "output_text": text,
        "usage": {
            "input_tokens": chat_response.usage.prompt_tokens,
            "output_tokens": chat_response.usage.completion_tokens,
            "total_tokens": chat_response.usage.total_tokens,
        },
    }


async def _stream_responses_sse(
    chunks: AsyncIterator[ChatCompletionChunk],
    *,
    model: str,
    raw_rest: bool,
) -> AsyncIterator[bytes]:
    response_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    created_at = int(time.time())
    text_parts: list[str] = []

    created_payload = {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": model,
            "output": [],
        },
    }
    yield _as_sse_data(created_payload, raw_rest)
    yield _as_sse_data(
        {
            "type": "response.in_progress",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "model": model,
                "output": [],
            },
        },
        raw_rest,
    )
    yield _as_sse_data(
        {
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": message_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        raw_rest,
    )
    yield _as_sse_data(
        {
            "type": "response.content_part.added",
            "response_id": response_id,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        raw_rest,
    )

    finish_reason = "stop"
    async for chunk in chunks:
        for choice in chunk.choices:
            if choice.delta.content:
                delta = choice.delta.content
                text_parts.append(delta)
                yield _as_sse_data(
                    {
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta,
                    },
                    raw_rest,
                )
            if choice.finish_reason:
                finish_reason = choice.finish_reason

    final_text = "".join(text_parts)
    yield _as_sse_data(
        {
            "type": "response.output_text.done",
            "response_id": response_id,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "text": final_text,
        },
        raw_rest,
    )
    yield _as_sse_data(
        {
            "type": "response.content_part.done",
            "response_id": response_id,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": final_text, "annotations": []},
        },
        raw_rest,
    )
    yield _as_sse_data(
        {
            "type": "response.output_item.done",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": final_text, "annotations": []}],
            },
        },
        raw_rest,
    )
    yield _as_sse_data(
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "completed",
                "model": model,
                "output": [
                    {
                        "id": message_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": final_text, "annotations": []}],
                    }
                ],
                "output_text": final_text,
                "finish_reason": finish_reason,
            },
        },
        raw_rest,
    )
    if raw_rest:
        logger.debug("[REST <- STREAM] [DONE]")
    yield b"data: [DONE]\n\n"


def _anthropic_payload_to_chat_request(payload: dict[str, Any], provider_name: str) -> ChatCompletionRequest:
    model = str(payload.get("model") or provider_name)
    stream = bool(payload.get("stream", True))
    max_tokens_raw = payload.get("max_tokens")
    max_tokens: int | None = max_tokens_raw if isinstance(max_tokens_raw, int) else None

    messages: list[ChatMessage] = []
    system = payload.get("system")
    if isinstance(system, str) and system.strip():
        messages.append(ChatMessage(role="system", content=system))
    elif isinstance(system, list):
        system_text = _content_to_text(system)
        if system_text:
            messages.append(ChatMessage(role="system", content=system_text))

    anthropic_messages = payload.get("messages")
    if isinstance(anthropic_messages, list):
        for item in anthropic_messages:
            if not isinstance(item, dict):
                continue
            role = _normalize_role(item.get("role", "user"))
            content_text = _content_to_text(item.get("content"))
            if content_text:
                messages.append(ChatMessage(role=role, content=content_text))

    if not messages:
        raise HTTPException(status_code=400, detail="Anthropic request must include at least one message")

    return ChatCompletionRequest(
        model=model,
        messages=messages,
        stream=stream,
        max_tokens=max_tokens,
    )


def _anthropic_json_from_chat_response(chat_response: ChatCompletionResponse) -> dict[str, Any]:
    text = chat_response.choices[0].message.content or ""
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": chat_response.model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": _anthropic_stop_reason(chat_response.choices[0].finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": chat_response.usage.prompt_tokens,
            "output_tokens": chat_response.usage.completion_tokens,
        },
    }


def _anthropic_stop_reason(finish_reason: str) -> str:
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    return "end_turn"


async def _stream_anthropic_sse(
    chunks: AsyncIterator[ChatCompletionChunk],
    *,
    model: str,
    raw_rest: bool,
) -> AsyncIterator[bytes]:
    message_id = f"msg_{uuid.uuid4().hex}"
    text_parts: list[str] = []
    finish_reason = "stop"

    message_start = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    yield _as_sse_event("message_start", message_start, raw_rest)
    yield _as_sse_event(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        raw_rest,
    )

    async for chunk in chunks:
        for choice in chunk.choices:
            if choice.delta.content:
                delta = choice.delta.content
                text_parts.append(delta)
                yield _as_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": delta},
                    },
                    raw_rest,
                )
            if choice.finish_reason:
                finish_reason = choice.finish_reason

    yield _as_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}, raw_rest)
    yield _as_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": _anthropic_stop_reason(finish_reason), "stop_sequence": None},
            "usage": {"output_tokens": len("".join(text_parts)) // 4},
        },
        raw_rest,
    )
    yield _as_sse_event("message_stop", {"type": "message_stop"}, raw_rest)


def _as_sse_data(payload: dict[str, Any], raw_rest: bool) -> bytes:
    payload_json = json.dumps(payload, ensure_ascii=True)
    if raw_rest:
        logger.debug("[REST <- STREAM] %s", payload_json)
    return f"data: {payload_json}\n\n".encode()


def _as_sse_event(event: str, payload: dict[str, Any], raw_rest: bool) -> bytes:
    payload_json = json.dumps(payload, ensure_ascii=True)
    if raw_rest:
        logger.debug("[REST <- STREAM] event=%s payload=%s", event, payload_json)
    return f"event: {event}\ndata: {payload_json}\n\n".encode()


async def _anthropic_messages_response(
    *,
    provider: BaseProvider,
    payload: dict[str, Any],
    provider_name: str,
    raw_rest: bool,
) -> Response:
    chat_request = _anthropic_payload_to_chat_request(payload, provider_name)
    if chat_request.stream:
        return StreamingResponse(
            _stream_anthropic_sse(
                provider.chat_completion_stream(chat_request),
                model=chat_request.model,
                raw_rest=raw_rest,
            ),
            media_type="text/event-stream",
        )

    chat_response = await provider.chat_completion(chat_request)
    return JSONResponse(content=_anthropic_json_from_chat_response(chat_response))


async def _stream_sse(chunks: AsyncIterator[ChatCompletionChunk], raw_rest: bool) -> AsyncIterator[bytes]:
    async for chunk in chunks:
        payload = chunk.model_dump_json(exclude_none=True)
        line = f"data: {payload}\n\n"
        if raw_rest:
            logger.debug("[REST <- STREAM] %s", payload)
        yield line.encode("utf-8")
    if raw_rest:
        logger.debug("[REST <- STREAM] [DONE]")
    yield b"data: [DONE]\n\n"


def _provider_from_path(path: str) -> str:
    """Extract provider name from URL path like /api/v1/{provider}/openai/..."""
    parts = path.strip("/").split("/")
    if len(parts) >= 3 and parts[0] == "api" and parts[1] == "v1":
        candidate = parts[2]
        if candidate not in ("providers", "config", "ws"):
            return candidate
    return "server"
