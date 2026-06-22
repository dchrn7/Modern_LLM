from __future__ import annotations

import base64
import io

import requests
from PIL import Image

DEFAULT_TIMEOUT_S = 180


def _encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_api(
    api_url: str,
    message: str,
    image: Image.Image | None = None,
    temperature: float = 0.7,
    max_turns: int = 8,
    max_new_tokens: int = 256,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> tuple[str, int]:
    """POST messages (+ optional image) to /chat and return (reply, generation_time_ms)."""
    payload: dict = {
        "messages": message,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
    }
    if image is not None:
        payload["image_base64"] = _encode_image(image)

    resp = requests.post(f"{api_url.rstrip('/')}/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["reply"], int(data["generation_time_ms"])
