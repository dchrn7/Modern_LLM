"""Dataset classes. PROVIDED — do not modify.

Three classes are available:
  • CauldronDataset  — streams from The Cauldron (HuggingFaceM4/the_cauldron)
  • FlickrDataset    — streams from Flickr30k (AnyModal/flickr30k)
  • DummyDataset     — synthetic data for unit tests (no download needed)

Both real datasets MUST be pre-downloaded and saved with dataset.save_to_disk()
by an admin (see prepare_datasets.py). Users then set TrainConfig.dataset_local_path
to that directory, which avoids HuggingFace's lock-file problem on shared filesystems.

Loading logic (in train.py):
    if train_cfg.dataset_local_path:
        raw = load_from_disk(train_cfg.dataset_local_path)   # lock-free Arrow
        split = raw["train"]
    else:
        split = load_dataset(train_cfg.dataset_path, train_cfg.dataset_name,
                             streaming=True)["train"]
    dataset = CauldronDataset(split, tokenizer, image_processor, cfg)
"""

import logging
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from data.processors import get_image_string


# ──────────────────────────────────────────────────────────────────────────────
class CauldronDataset(IterableDataset):
    """Wrap one split (e.g. raw["train"]) of a Cauldron subset.

    Expected row schema:
        images: list of PIL.Image  (we take only the first)
        texts:  list of {"user": str, "assistant": str}

    Returns dicts with keys: input_ids, attention_mask, labels, pixel_values.
    Returns None for rows that cannot be converted (caller / collator skips them).
    """

    def __init__(self, dataset, tokenizer, image_processor, cfg):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.cfg = cfg
        self.image_string = get_image_string(
            cfg.projector.image_token_length, cfg.image_token
        )

    def _build_messages(self, item):
        """Convert a Cauldron row into a chat transcript."""
        messages = []
        texts = item.get("texts", [])
        if not texts:
            return []
        for turn in texts:
            messages.append({"role": "user",      "content": turn.get("user", "")})
            messages.append({"role": "assistant", "content": turn.get("assistant", "")})
        if not messages:
            return []
        # Prepend image placeholder tokens to the first user turn
        for msg in messages:
            if self.tokenizer.image_token in msg["content"]:
                logging.warning("Removed unexpected image token from sample text.")
                msg["content"] = msg["content"].replace(self.tokenizer.image_token, "")
        messages[0]["content"] = self.image_string + messages[0]["content"]
        return messages

    def _tokenize(self, messages):
        """Tokenize conversation; label only assistant tokens (others → -100)."""
        full = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_special_tokens=False, return_dict=True
        )
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        labels = [-100] * len(input_ids)

        for idx, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            prefix_ids = self.tokenizer.apply_chat_template(
                messages[:idx], tokenize=True, add_generation_prompt=True, return_dict=True
            )["input_ids"]
            full_ids = self.tokenizer.apply_chat_template(
                messages[:idx + 1], tokenize=True, add_special_tokens=False, return_dict=True
            )["input_ids"]
            start, end = len(prefix_ids), len(full_ids)
            if end > start:
                labels[start:end] = full_ids[start:end]

        return (
            torch.tensor(input_ids),
            torch.tensor(attention_mask),
            torch.tensor(labels),
        )

    def _process_image(self, image):
        if not isinstance(image, Image.Image):
            return None
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.image_processor(image)

    def __iter__(self):
        for item in self.dataset:
            try:
                images = item.get("images") or []
                if not images:
                    continue
                pixel_values = self._process_image(images[0])
                if pixel_values is None:
                    continue
                messages = self._build_messages(item)
                if not messages:
                    continue
                input_ids, attention_mask, labels = self._tokenize(messages)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "pixel_values": pixel_values,
                }
            except Exception as e:
                logging.warning(f"Skipping sample: {e}")
                continue


# ──────────────────────────────────────────────────────────────────────────────
class FlickrDataset(IterableDataset):
    """Wrap one split of AnyModal/flickr30k.

    Expected row schema:
        image:             PIL.Image
        original_alt_text: str  (preferred) or alt_text: str

    Each sample becomes a simple "Describe the image." → caption conversation.
    """

    CAPTION_PROMPT = "Describe the image."

    def __init__(self, dataset, tokenizer, image_processor, cfg):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.cfg = cfg
        self.image_string = get_image_string(
            cfg.projector.image_token_length, cfg.image_token
        )

    def _get_caption(self, item):
        captions = item.get("original_alt_text") or item.get("alt_text") or []
        if isinstance(captions, str):
            captions = [captions]
        for c in captions:
            if c and c.strip():
                return c.strip()
        return None

    def _build_messages(self, caption):
        return [
            {"role": "user",      "content": self.image_string + self.CAPTION_PROMPT},
            {"role": "assistant", "content": caption},
        ]

    def _tokenize(self, messages):
        full = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_special_tokens=False, return_dict=True
        )
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        labels = [-100] * len(input_ids)

        for idx, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            prefix_ids = self.tokenizer.apply_chat_template(
                messages[:idx], tokenize=True, add_generation_prompt=True, return_dict=True
            )["input_ids"]
            full_ids = self.tokenizer.apply_chat_template(
                messages[:idx + 1], tokenize=True, add_special_tokens=False, return_dict=True
            )["input_ids"]
            start, end = len(prefix_ids), len(full_ids)
            if end > start:
                labels[start:end] = full_ids[start:end]

        return (
            torch.tensor(input_ids),
            torch.tensor(attention_mask),
            torch.tensor(labels),
        )

    def __iter__(self):
        for item in self.dataset:
            try:
                image = item.get("image")
                if not isinstance(image, Image.Image):
                    continue
                if image.mode != "RGB":
                    image = image.convert("RGB")
                pixel_values = self.image_processor(image)

                caption = self._get_caption(item)
                if caption is None:
                    continue

                messages = self._build_messages(caption)
                input_ids, attention_mask, labels = self._tokenize(messages)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "pixel_values": pixel_values,
                }
            except Exception as e:
                logging.warning(f"Skipping Flickr sample: {e}")
                continue


# ──────────────────────────────────────────────────────────────────────────────
class DummyDataset(Dataset):
    """Synthetic dataset that requires no download — used by the test suite.

    Each sample has a random 512×512 image tensor and a fixed Q&A conversation.
    The labels are constructed so that only the answer tokens contribute to loss.

    Example:
        dataset = DummyDataset(size=64, seq_len=32)
        sample  = dataset[0]
        # sample["input_ids"].shape == torch.Size([32])
    """

    MESSAGES = [
        {"role": "user",      "content": "What colour is the sky?"},
        {"role": "assistant", "content": "The sky is blue."},
    ]

    def __init__(self, size: int = 64, seq_len: int = 64, img_size: int = 512, seed: int = 0):
        self.size = size
        self.seq_len = seq_len
        self.img_size = img_size
        self.rng = random.Random(seed)
        # Fixed token ids for fast tests (no tokenizer needed)
        self._input_ids = torch.randint(0, 49153, (seq_len,), generator=torch.Generator().manual_seed(seed))
        self._attention_mask = torch.ones(seq_len, dtype=torch.long)
        # Only last third of tokens are "answer" tokens
        self._labels = self._input_ids.clone()
        answer_start = seq_len * 2 // 3
        self._labels[:answer_start] = -100

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        g = torch.Generator()
        g.manual_seed(idx)
        pixel_values = torch.randn(3, self.img_size, self.img_size, generator=g)
        return {
            "input_ids":      self._input_ids.clone(),
            "attention_mask": self._attention_mask.clone(),
            "labels":         self._labels.clone(),
            "pixel_values":   pixel_values,
        }
