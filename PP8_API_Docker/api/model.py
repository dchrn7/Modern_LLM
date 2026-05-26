from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

logger = logging.getLogger(__name__)

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

_processor: AutoProcessor | None = None
_model: AutoModelForVision2Seq | None = None
_device: str = "cpu"


def load_model() -> None:
    global _processor, _model, _device
    if _model is not None:
        return

    if torch.cuda.is_available():
        _device = "cuda"
        dtype = torch.bfloat16
    else:
        _device = "cpu"
        dtype = torch.float32

    logger.info("Loading %s on %s (dtype=%s)", MODEL_ID, _device, dtype)
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=dtype).to(_device)
    _model.eval()
    logger.info("Model loaded")


def generate(
    messages: list[dict],
    image: Image.Image | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate the next assistant reply for a conversation.

    `messages` uses the `{"role": ..., "content": str}` format — same shape the
    notebook taught and the same shape the OpenAI chat API exposes. The
    SmolVLM-specific content-block conversion happens here and stays here.
    """
    if _model is None or _processor is None:
        raise RuntimeError("Model not loaded; call load_model() first")

    has_image = image is not None
    last_user_idx = max(
        (i for i, m in enumerate(messages) if m["role"] == "user"),
        default=-1,
    )

    # SmolVLM-specific: convert {role, content: str} → SmolVLM content-block format.
    formatted: list[dict] = []
    for i, m in enumerate(messages):
        if m["role"] == "user" and i == last_user_idx and has_image:
            content = [{"type": "image"}, {"type": "text", "text": m["content"]}]
        else:
            content = [{"type": "text", "text": m["content"]}]
        formatted.append({"role": m["role"], "content": content})

    prompt = _processor.apply_chat_template(formatted, add_generation_prompt=True)
    inputs = _processor(
        text=prompt,
        images=[image] if has_image else None,
        return_tensors="pt",
    ).to(_device)

    do_sample = temperature > 0.0
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, input_len:]
    return _processor.decode(new_tokens, skip_special_tokens=True).strip()
