"""Vision-Language Model — student skeleton.

This file wires together the four components you have implemented:
    1. ViT              — encodes the image into patch tokens
    2. ModalityProjector — maps ViT tokens to LM embedding space
    3. LanguageModel    — processes the merged sequence
    4. Tokenizer        — converts text ↔ token ids

The key idea: image placeholder tokens (<|image|> × 64) in the prompt are
REPLACED by the projected image embeddings before being fed to the LM.

                  input_ids  [B, T]
                      │
              token_embedding   → token_embd  [B, T, 960]
                      │
  pixel_values [B,3,512,512]
      │ vision_encoder                        ← your ViT
      │   [B, 1024, 768]
      │ MP (modality projector)               ← your ModalityProjector
      │   [B, 64, 960]
      │
  _replace_img_tokens_with_embd              ← PROVIDED
      │   token_embd with 64 slots replaced
      │
  decoder.forward(token_embd)                ← your LanguageModel
      │   hidden  [B, T, 960]
      │
  decoder.head(hidden)  (only if targets ≠ None)
      │   logits  [B, T, vocab_size]
      │
  cross_entropy loss
"""

import json
import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from models.config import VLMConfig
from models.vision_transformer import ViT  # noqa: F401
from models.language_model import LanguageModel  # noqa: F401
from models.modality_projector import ModalityProjector  # noqa: F401
from data.processors import get_tokenizer  # noqa: F401


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("inf")):
    """Nucleus (top-p) and top-k filtering for token sampling. PROVIDED."""
    if top_k > 0:
        topk_vals = torch.topk(logits, top_k)[0]
        logits[logits < topk_vals[..., [-1]]] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_idx_to_remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_idx_to_remove] = filter_value
        logits.scatter_(-1, sorted_idx, sorted_logits)
    return logits


# ──────────────────────────────────────────────────────────────────────────────
class VisionLanguageModel(nn.Module):

    def __init__(self, cfg: VLMConfig, load_backbone: bool = True):
        """Wire together the four components.

        Args:
            cfg:           VLMConfig with all architecture hyperparameters.
            load_backbone: If True, download pretrained SigLIP2 + SmolLM2 weights.
                           Set to False for unit tests (avoids slow downloads).
        """
        super().__init__()
        self.cfg = cfg

        self.vision_encoder = ViT.from_pretrained(cfg.vit) if load_backbone else ViT(self.cfg.vit)   # the ViT image encoder
        #   if load_backbone: use ViT.from_pretrained(cfg.vit)
        #   else:             use ViT(cfg.vit)
        self.decoder = LanguageModel.from_pretrained(cfg.lm) if load_backbone else LanguageModel(cfg.lm)          # the causal language model
        #   if load_backbone: use LanguageModel.from_pretrained(cfg.lm)
        #   else:             use LanguageModel(cfg.lm)
        self.MP = ModalityProjector(cfg)               # the ModalityProjector
        self.tokenizer = get_tokenizer(cfg.lm.tokenizer)        # the tokenizer (use get_tokenizer)

        # raise NotImplementedError

    # ── PROVIDED — image token replacement ───────────────────────────────────
    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        """Replace every <|image|> placeholder with its visual embedding.

        There are exactly mp_image_token_length=64 such placeholders per image.
        This method handles batches where different samples may have a different
        number of images by operating through a flat boolean mask.

        Args:
            input_ids:   [B, T]           token ids containing image_token_id
            token_embd:  [B, T, D_lm]     full sequence embeddings
            image_embd:  [N×64, D_lm]     projected visual tokens (N = total images)
        Returns:
            [B, T, D_lm]  with image positions overwritten
        """
        updated = token_embd.clone()
        mask = (input_ids == self.tokenizer.image_token_id)
        updated[mask] = image_embd.view(-1, image_embd.size(-1)).to(updated.dtype)
        return updated

    # ── PROVIDED — image pre-processing ──────────────────────────────────────
    def _process_images(self, pixel_values, device):
        """Accept either a [B,C,H,W] tensor or a list and move to device."""
        if isinstance(pixel_values, list):
            pixel_values = torch.stack(pixel_values, dim=0)
        return pixel_values.to(device)

    # ── STUDENT WORK — forward pass ───────────────────────────────────────────
    def forward(self, input_ids, pixel_values, attention_mask=None, targets=None):
        """Multimodal forward pass (training).

        Args:
            input_ids:      [B, T]         token ids; <|image|> tokens are placeholders
            pixel_values:   [B, 3, 512, 512]  already preprocessed images
            attention_mask: [B, T]         1=attend, 0=pad  (optional)
            targets:        [B, T]         ground-truth labels with -100 for ignored
                                           positions (optional; if None, no loss)
        Returns:
            logits: [B, T, lm_vocab_size]  (or hidden [B, T, D_lm] if targets=None)
            loss:   scalar or None

        Step-by-step guide:
        ──────────────────
        TODO 1 — Embed the input token ids into the LM's embedding space.
                 Output: [B, T, 960]

        TODO 2 — Pre-process the images and run them through the vision
                 encoder.
                 Output: [B, 1024, 768]

        TODO 3 — Project the visual features to the LM's embedding space
                 via the modality projector.
                 Output: [B, 64, 960]

        TODO 4 — Replace the image placeholder tokens in the sequence with
                 the projected visual embeddings.

        TODO 5 — Run the merged embedding sequence through the language
                 model.
                 Output: hidden [B, T, 960]

        TODO 6 — If targets are provided: apply the language model head to
                 get logits, compute cross-entropy loss (ignore_index=-100),
                 and return (logits, loss).

        TODO 7 — If no targets: return (hidden, None) for generation.
        """
        # TODO 1
        token_embd = self.decoder.token_embedding(input_ids)

        # TODO 2
        pixel_values = self._process_images(pixel_values, input_ids.device)
        image_features = self.vision_encoder(pixel_values)

        # TODO 3
        image_embd = self.MP(image_features)

        # TODO 4
        inputs_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        # TODO 5
        hidden, _ = self.decoder(inputs_embd, attention_mask=attention_mask)

        if targets is not None:
            # TODO 6
            logits = self.decoder.head(hidden)

            # logits[:, i] predicts targets[:, i+1] → drop last logit, drop first target
            shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
            shift_targets = targets[:, 1:].contiguous()      # [B, T-1]

            loss = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100
            )

            return logits, loss
        else:
            # TODO 7
            return hidden, None


        # raise NotImplementedError

    # ── PROVIDED — autoregressive generation ─────────────────────────────────
    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        max_new_tokens=64,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        greedy=False,
    ):
        """Autoregressively decode text conditioned on image + prompt. PROVIDED."""
        token_embd = self.decoder.token_embedding(input_ids)
        if pixel_values is not None:
            images = self._process_images(pixel_values, input_ids.device)
            image_feats = self.vision_encoder(images)
            image_embd = self.MP(image_feats)
            # On remplace les placeholders <|image|> par les caractéristiques de l'image
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)
        current_seq_len = token_embd.size(1)
        
        batch_size = input_ids.size(0)

        prefill_out, kv_cache = self.decoder(
            token_embd, attention_mask=attention_mask, kv_cache=None, start_pos=0
        )
        current_logits = self.decoder.head(prefill_out[:, -1, :])

        generated = []
        for _ in range(max_new_tokens):
            if greedy:
                next_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            generated.append(next_id)
            next_embd = self.decoder.token_embedding(next_id)
            start_pos = current_seq_len
            current_seq_len += 1

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)], dim=1
                )

            decode_out, kv_cache = self.decoder(
                next_embd, attention_mask=attention_mask, kv_cache=kv_cache, start_pos=start_pos
            )
            current_logits = self.decoder.head(decode_out[:, -1, :])

        if not generated:
            return torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)

        generated_ids = torch.cat(generated, dim=1)

        # Truncate at first EOS
        if self.tokenizer.eos_token_id is not None:
            seq_len = generated_ids.size(1)
            eos_mask = generated_ids == self.tokenizer.eos_token_id
            col_idx = torch.arange(seq_len, device=generated_ids.device)
            masked = torch.where(eos_mask, col_idx.unsqueeze(0).expand_as(generated_ids), seq_len + 1)
            first_eos = torch.clamp(masked.min(dim=1).values, max=seq_len)
            replace = col_idx.unsqueeze(0).expand_as(generated_ids) > first_eos.unsqueeze(1)
            generated_ids[replace] = self.tokenizer.eos_token_id

        return generated_ids

    # ── PROVIDED — save / load checkpoints ───────────────────────────────────
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(asdict(self.cfg), f, indent=4)
        save_model(self, os.path.join(save_directory, "model.safetensors"))
        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, path: str, revision: Optional[str] = None):
        """Load a trained VLM checkpoint from a local directory."""
        config_path = os.path.join(path, "config.json")
        weights_path = os.path.join(path, "model.safetensors")
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise ValueError(f"Expected config.json and model.safetensors in {path}")
        with open(config_path) as f:
            cfg = VLMConfig.from_dict(json.load(f))
        model = cls(cfg, load_backbone=False)
        load_model(model, weights_path)
        return model
