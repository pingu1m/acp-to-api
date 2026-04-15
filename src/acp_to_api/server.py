from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Callable, Awaitable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from acp_to_api.config import AppConfig
from acp_to_api.openai_models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    OpenAIModel,
    OpenAIModelsResponse,
)
from acp_to_api.providers import BaseProvider, CursorACPProvider

logger = logging.getLogger(__name__)


def _build_providers(config: AppConfig) -> dict[str, BaseProvider]:
    return {
        name: CursorACPProvider(provider_cfg, raw_acp=config.raw_acp)
        for name, provider_cfg in config.providers.items()
    }


def create_app(config: AppConfig) -> FastAPI:
    providers = _build_providers(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        for provider in providers.values():
            await provider.startup()
        try:
            yield
        finally:
            for provider in providers.values():
                await provider.shutdown()

    app = FastAPI(title="acp-to-api", version="0.1.0", lifespan=lifespan)
    app.state.providers = providers
    app.state.raw_rest = config.raw_rest

    if config.raw_rest:
        app.middleware("http")(_raw_rest_middleware)

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

    return app


def _get_provider(app: FastAPI, provider_name: str) -> BaseProvider:
    provider = app.state.providers.get(provider_name)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_name}")
    return provider


async def _stream_sse(
    chunks: AsyncIterator[ChatCompletionChunk], raw_rest: bool
) -> AsyncIterator[bytes]:
    async for chunk in chunks:
        payload = chunk.model_dump_json(exclude_none=True)
        line = f"data: {payload}\n\n"
        if raw_rest:
            logger.debug("[REST <- STREAM] %s", payload)
        yield line.encode("utf-8")
    if raw_rest:
        logger.debug("[REST <- STREAM] [DONE]")
    yield b"data: [DONE]\n\n"


async def _raw_rest_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    body = await request.body()
    body_text = body.decode("utf-8", errors="replace") if body else ""
    logger.debug("[REST ->] %s %s %s", request.method, request.url.path, body_text)

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
    logger.debug("[REST <-] %s %s", response.status_code, resp_text)

    headers = dict(response.headers)
    return Response(
        content=content,
        status_code=response.status_code,
        headers=headers,
        media_type=response.media_type,
    )
