from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


MessageRole = Literal["system", "developer", "user", "assistant", "tool"]


class ImageUrlPart(BaseModel):
    url: str
    detail: Optional[str] = None


class MessagePartText(BaseModel):
    type: Literal["text"]
    text: str


class MessagePartImageUrl(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrlPart


MessagePart = Union[MessagePartText, MessagePartImageUrl]


class ChatMessage(BaseModel):
    role: MessageRole
    content: Optional[Union[str, list[MessagePart]]] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None


class ChatCompletionResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionResponseMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class DeltaMessage(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class OpenAIModel(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "acp-to-api"


class OpenAIModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIModel] = Field(default_factory=list)


def message_content_to_text(message: ChatMessage) -> str:
    if message.content is None:
        return ""
    if isinstance(message.content, str):
        return message.content

    parts: list[str] = []
    for part in message.content:
        if isinstance(part, MessagePartText):
            parts.append(part.text)
        elif isinstance(part, MessagePartImageUrl):
            parts.append(f"[image] {part.image_url.url}")
    return "\n".join(parts)
