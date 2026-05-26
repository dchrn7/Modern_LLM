from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    image_base64: str | None = None
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_turns: int = Field(default=8, ge=1, le=32)


class ChatResponse(BaseModel):
    reply: str
    generation_time_ms: int
