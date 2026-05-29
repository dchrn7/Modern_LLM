"""Batch collation. PROVIDED — do not modify."""

import torch
from torch.utils.data import default_collate


class VQACollator:
    """Collate variable-length multimodal samples into a padded batch.

    Pads input_ids and attention_mask to the longest sequence in the batch.
    Pads labels with -100 (ignored by cross-entropy).
    Stacks pixel_values into [B, 3, H, W].
    Drops None samples and samples exceeding max_length.
    """

    def __init__(self, tokenizer, max_length: int = 2048):
        self.pad_id = tokenizer.pad_token_id
        self.max_length = max_length

    def __call__(self, batch):
        # Filter bad samples
        batch = [s for s in batch if s is not None]
        batch = [s for s in batch if len(s["input_ids"]) <= self.max_length]
        if not batch:
            return None

        max_len = max(len(s["input_ids"]) for s in batch)

        input_ids_list, attn_list, labels_list, pixel_list = [], [], [], []
        for s in batch:
            T = len(s["input_ids"])
            pad = max_len - T
            input_ids_list.append(
                torch.cat([s["input_ids"], torch.full((pad,), self.pad_id, dtype=torch.long)])
            )
            attn_list.append(
                torch.cat([s["attention_mask"], torch.zeros(pad, dtype=torch.long)])
            )
            labels_list.append(
                torch.cat([s["labels"], torch.full((pad,), -100, dtype=torch.long)])
            )
            pixel_list.append(s["pixel_values"])

        return {
            "input_ids":      torch.stack(input_ids_list),    # [B, T]
            "attention_mask": torch.stack(attn_list),          # [B, T]
            "labels":         torch.stack(labels_list),        # [B, T]
            "pixel_values":   torch.stack(pixel_list),         # [B, 3, H, W]
        }
