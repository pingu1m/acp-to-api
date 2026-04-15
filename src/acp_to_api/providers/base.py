from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from acp_to_api.openai_models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class BaseProvider(ABC):
    name: str

    @abstractmethod
    async def startup(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    def list_models(self) -> list[str]: ...

    @abstractmethod
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...

    @abstractmethod
    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]: ...
