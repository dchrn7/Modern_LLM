from __future__ import annotations

import base64
import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from PIL import Image

import model
from schemas import ChatMessage, ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load_model()
    yield


app = FastAPI(title="VLM Chatbot API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def truncate_history(messages: list[ChatMessage], max_turns: int) -> list[ChatMessage]:
    """Keep any leading system message + the last 2*max_turns user/assistant entries.

    Lives in the API layer because it's about HTTP-level context-size bookkeeping,
    not about the model itself.
    """
    system_prefix = [m for m in messages[:1] if m.role == "system"]
    rest = messages[len(system_prefix):]
    kept_tail = rest[-(2 * max_turns):]
    return system_prefix + kept_tail


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")

    image: Image.Image | None = None
    if req.image_base64:
        try:
            raw = base64.b64decode(req.image_base64)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid image_base64: {e}") from e

    trimmed = truncate_history(req.messages, req.max_turns)
    messages_for_model = [{"role": m.role, "content": m.content} for m in trimmed]

    start = time.perf_counter()
    reply = model.generate(
        messages=messages_for_model,
        image=image,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info("chat: %d messages, image=%s, %d ms", len(trimmed), image is not None, elapsed_ms)

    return ChatResponse(reply=reply, generation_time_ms=elapsed_ms)
