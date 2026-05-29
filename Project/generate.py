"""Demo inference script. PROVIDED — do not modify.

Load a trained checkpoint and generate answers for an image + text prompt.

Usage:
    # From a local checkpoint saved by train.py
    python generate.py --checkpoint checkpoints/best_step5000 --image path/to/image.jpg

    # With a custom prompt
    python generate.py --checkpoint checkpoints/best_step5000 \\
        --image path/to/image.jpg \\
        --prompt "Describe this image in detail."

    # Generate multiple samples
    python generate.py --checkpoint checkpoints/best_step5000 \\
        --image path/to/image.jpg --n 3
"""

import argparse

import torch
from PIL import Image

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string


def parse_args():
    p = argparse.ArgumentParser(description="VLM inference demo")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a local checkpoint directory (output of train.py).")
    p.add_argument("--image", type=str, required=True,
                   help="Path to the input image.")
    p.add_argument("--prompt", type=str, default="Describe the image.",
                   help="Text prompt.")
    p.add_argument("--n", type=int, default=1,
                   help="Number of generations.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--greedy", action="store_true",
                   help="Use greedy decoding instead of sampling.")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=0.8)
    return p.parse_args()


def main():
    args = parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    print(f"Loading checkpoint from {args.checkpoint} …")
    model = VisionLanguageModel.from_pretrained(args.checkpoint).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded — {n_params:,} parameters")

    cfg = model.cfg
    tokenizer = get_tokenizer(cfg.lm.tokenizer, cfg.image_token)
    image_processor = get_image_processor(cfg.vit.img_size)

    # Process image
    img = Image.open(args.image).convert("RGB")
    pixel_values = image_processor(img).unsqueeze(0).to(device)

    # Build prompt with image placeholder tokens
    image_string = get_image_string(
        cfg.projector.image_token_length, cfg.image_token
    )
    messages = [{"role": "user", "content": image_string + args.prompt}]
    encoded = tokenizer.apply_chat_template(
        [messages], tokenize=True, add_generation_prompt=True
    )
    input_ids = torch.tensor(encoded).to(device)    # [1, T]
    attention_mask = torch.ones_like(input_ids)

    print(f"\nPrompt: {args.prompt}\n")
    for i in range(args.n):
        with torch.no_grad():
            gen = model.generate(
                input_ids, pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                greedy=args.greedy,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
            )
        text = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        print(f"[{i+1}] {text}")


if __name__ == "__main__":
    main()
