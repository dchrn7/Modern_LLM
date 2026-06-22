from __future__ import annotations

import logging
import os
import torch
from PIL import Image
from models.config import TrainConfig
from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string

logger = logging.getLogger(__name__)
_tokenizer = None
_image_processor = None
_device: str = "cpu"
_model = None
vlm_cfg = None
checkpoint_path = TrainConfig.checkpoint_dir

def load_model() -> None:
    global _image_processor, _model, _device, _tokenizer, vlm_cfg
    if _model is not None:
        return

    if torch.cuda.is_available():
        _device = "cuda"
        dtype = torch.bfloat16
    else:
        _device = "cpu"
        dtype = torch.float32

    logger.info("Loading %s on %s (dtype=%s)",checkpoint_path, _device, dtype)
    
    _model = VisionLanguageModel.from_pretrained(checkpoint_path).to(_device).to(dtype)
    vlm_cfg = _model.cfg
    _tokenizer = get_tokenizer(vlm_cfg.lm.tokenizer, vlm_cfg.image_token)
    _image_processor = get_image_processor(vlm_cfg.vit.img_size)
    _model.eval()
    logger.info("Model loaded")



def generate(message, image=None, max_new_tokens: int = 128, greedy: bool = False, top_k: int = 50, top_p: float = 0.9, temperature: float = 1.0) -> str:
    load_model()
    
    # Par défaut (cas où il n'y a que du texte)
    pixel_values = None
    content = message
    
    # S'il y a une image, on la traite et on adapte le contenu
    if image is not None:
        img = image.convert("RGB") if image.mode != "RGB" else image
        dtype = next(_model.parameters()).dtype
        pixel_values = _image_processor(img).unsqueeze(0).to(_device).to(dtype)

        # Build prompt with image placeholder tokens
        image_string = get_image_string(
            vlm_cfg.projector.image_token_length, vlm_cfg.image_token
        )
        content = image_string + message

    # Construction du format conversationnel avec le contenu adapté
    messages = [{"role": "user", "content": content}]
    encoded = _tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    
    input_ids = torch.tensor(encoded).to(_device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        gen = _model.generate(
            input_ids, 
            pixel_values, # Sera passé en None si aucune image n'est fournie
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            greedy=greedy,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        
    text = _tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return text