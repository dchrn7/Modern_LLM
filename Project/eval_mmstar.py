"""Evaluate a trained VLM checkpoint on MMStar validation.

This script is intended for student-visible validation only.  Teachers can
reuse the same evaluator with a private dataset path for final evaluation.

Usage:
    python eval_mmstar.py \
        --checkpoint checkpoints/best_step5000 \
        --dataset_local_path /shared/datasets/mmstar \
        --split val
"""

import argparse
import json
import os
import re
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm

from data.processors import get_image_processor, get_image_string, get_tokenizer
from models.vision_language_model import VisionLanguageModel


MMSTAR_REPO = "Lin-Chen/MMStar"
VALID_ANSWER_LETTERS = ("A", "B", "C", "D")


def extract_answer_letter(text: str) -> str | None:
    """Return the first standalone A/B/C/D answer letter, if present."""
    if not text:
        return None
    match = re.search(r"\b([ABCD])\b", text.upper())
    if match is None:
        return None
    return match.group(1)


def normalize_mmstar_question(question: str) -> str:
    """Apply lightweight MMStar prompt formatting inspired by nanoVLM."""
    question = question or ""
    replacements = {
        "\nOptions:": "\nChoices:",
        "\nA. ": "\nChoices:\nA. ",
        "Please select the correct answer from the options above.": (
            "Answer with the letter directly."
        ),
        "Answer with the option's letter from the given choices directly": (
            "Answer with the letter directly."
        ),
    }
    for old, new in replacements.items():
        question = question.replace(old, new)
    if "Answer with the letter" not in question:
        question = question.rstrip() + "\nAnswer with the letter directly."
    return question


def build_mmstar_prompt(question: str, tokenizer, cfg) -> str:
    """Build the raw chat prompt used for MMStar generation."""
    image_string = get_image_string(
        cfg.projector.image_token_length, cfg.image_token
    )
    messages = [
        {
            "role": "user",
            "content": image_string + normalize_mmstar_question(question),
        }
    ]
    prompt = tokenizer.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=True
    )
    if isinstance(prompt, list):
        return prompt[0]
    return prompt


def get_mmstar_question(item: dict) -> str:
    for key in ("question", "query", "text", "problem"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError("MMStar sample does not contain a question-like field.")


def get_mmstar_answer(item: dict) -> str:
    for key in ("answer", "label", "gt_answer"):
        value = item.get(key)
        if value is None:
            continue
        answer = extract_answer_letter(str(value))
        if answer is not None:
            return answer
    raise KeyError("MMStar sample does not contain an answer-like field.")


def get_mmstar_image(item: dict) -> Image.Image:
    for key in ("image", "img", "decoded_image"):
        value = item.get(key)
        if isinstance(value, Image.Image):
            return value.convert("RGB")
    raise KeyError("MMStar sample does not contain a PIL image field.")


def get_mmstar_category(item: dict) -> str:
    for key in ("category", "l2_category", "subcategory", "discipline"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "unknown"


def load_mmstar_dataset(
    dataset_local_path: str | None,
    dataset_name: str,
    split: str,
):
    from datasets import load_dataset, load_from_disk

    if dataset_local_path:
        raw = load_from_disk(dataset_local_path)
        if split not in raw:
            available = ", ".join(raw.keys())
            raise ValueError(
                f"Split '{split}' not found in MMStar data. "
                f"Available splits: {available}"
            )
        return raw[split]

    return load_dataset(dataset_name, split=split)


def _encode_prompt(tokenizer, prompt: str, device: torch.device):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return input_ids, attention_mask


@torch.inference_mode()
def evaluate_mmstar(
    model,
    dataset,
    tokenizer,
    image_processor,
    device: torch.device,
    limit: int | None = None,
    max_new_tokens: int = 8,
    show_progress: bool = True,
) -> dict:
    total = 0
    correct = 0
    invalid = 0
    per_category = defaultdict(lambda: {"total": 0, "correct": 0})
    predictions = []

    if limit is not None:
        n_items = min(limit, len(dataset))
        iterable = (dataset[i] for i in range(n_items))
    else:
        n_items = len(dataset) if hasattr(dataset, "__len__") else None
        iterable = dataset

    for item in tqdm(iterable, total=n_items, disable=not show_progress):
        question = get_mmstar_question(item)
        gold = get_mmstar_answer(item)
        category = get_mmstar_category(item)

        prompt = build_mmstar_prompt(question, tokenizer, model.cfg)
        input_ids, attention_mask = _encode_prompt(tokenizer, prompt, device)

        image = get_mmstar_image(item)
        pixel_values = image_processor(image).unsqueeze(0).to(device)

        generated_ids = model.generate(
            input_ids,
            pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            greedy=True,
        )
        decoded = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        pred = extract_answer_letter(decoded)

        total += 1
        per_category[category]["total"] += 1
        if pred is None:
            invalid += 1
        elif pred == gold:
            correct += 1
            per_category[category]["correct"] += 1

        predictions.append(
            {
                "index": total - 1,
                "category": category,
                "gold": gold,
                "prediction": pred,
                "raw_prediction": decoded,
            }
        )

    category_metrics = {}
    for category, values in per_category.items():
        cat_total = values["total"]
        cat_correct = values["correct"]
        category_metrics[category] = {
            "accuracy": cat_correct / cat_total if cat_total else 0.0,
            "correct": cat_correct,
            "total": cat_total,
        }

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "invalid": invalid,
        "per_category": dict(sorted(category_metrics.items())),
        "predictions": predictions,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VLM checkpoint on MMStar validation."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_local_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default=MMSTAR_REPO)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--modality_projector_type", type=str, default="baseline")
    return parser.parse_args()


def main():
    args = parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    print(f"Loading checkpoint from {args.checkpoint}")
    model = VisionLanguageModel.from_pretrained(args.checkpoint, modality_projector_type=args.modality_projector_type).to(device)
    model.eval()

    tokenizer = get_tokenizer(model.cfg.lm.tokenizer, model.cfg.image_token)
    image_processor = get_image_processor(model.cfg.vit.img_size)
    dataset = load_mmstar_dataset(
        args.dataset_local_path or None, args.dataset_name, args.split
    )

    metrics = evaluate_mmstar(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )
    output = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset_local_path or args.dataset_name,
        "split": args.split,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "metrics": metrics,
    }

    print(
        "MMStar accuracy: "
        f"{metrics['accuracy']:.4f} "
        f"({metrics['correct']}/{metrics['total']}), "
        f"invalid={metrics['invalid']}"
    )

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
