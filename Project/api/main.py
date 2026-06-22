from __future__ import annotations

import base64
import io
import logging
import time
from api import model

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from PIL import Image
from api.schemas import ChatMessage, ChatRequest, ChatResponse

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

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages.strip():
        raise HTTPException(status_code=400, detail="messages must be non-empty")

    image: Image.Image | None = None
    if req.image_base64:
        try:
            raw = base64.b64decode(req.image_base64)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid image_base64: {e}") from e

    message = req.messages

    start = time.perf_counter()
    reply = model.generate(message, image, max_new_tokens=req.max_new_tokens, greedy=req.greedy, top_k=req.top_k, top_p=req.top_p, temperature=req.temperature)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info("chat: message length %d, image=%s, %d ms", len(message), image is not None, elapsed_ms)
    return ChatResponse(reply=reply, generation_time_ms=elapsed_ms)