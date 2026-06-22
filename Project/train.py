"""Training script for the VLM project.

Usage:
    python train.py --dataset_local_path /shared/datasets/the_cauldron/ai2d
    python train.py --dataset_type flickr \
        --dataset_local_path /shared/datasets/flickr30k
    python train.py --batch_size 1 --max_steps 100  # quick smoke test

The PROVIDED sections handle:
  * argument parsing and config override
  * data loading (CauldronDataset / FlickrDataset + DataLoader)
  * model construction
  * optimizer setup (3 parameter groups with different learning rates)
  * cosine LR schedule with warmup
  * evaluation loop
  * checkpoint saving

The STUDENT SECTION (clearly marked below) is the inner training loop body.
"""

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import fields

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.config import VLMConfig, TrainConfig
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
from data.collator import VQACollator

user = os.environ.get("USER")

# ── Cosine LR schedule with linear warmup ────────────────────────────────────
def get_lr(step: int, max_lr: float, max_steps: int) -> float:
    """Return the learning rate for a given step.

    Phase 1: linear ramp from 0 → max_lr over the first 3% of steps.
    Phase 2: cosine decay from max_lr → max_lr/10 over the remaining steps.
    """
    min_lr = max_lr * 0.1
    warmup_steps = max(1, int(max_steps * 0.03))
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (
        1.0 + math.cos(math.pi * decay)
    )


# =========================================
# We want to shuffle the dataset.
# We work with the index because
#   - Dataset is big
#   - We can skip the steps already done in the precedent jobs

# ── Data loading (PROVIDED) ───────────────────────────────────────────────────
def get_dataloaders(train_cfg: TrainConfig, vlm_cfg: VLMConfig, samples_consumed: int):
    from datasets import load_from_disk, concatenate_datasets

    if not train_cfg.dataset_local_path:
        raise ValueError(
            "dataset_local_path is required. "
            "Run prepare_datasets.py first, then set --dataset_local_path."
        )

    tokenizer = get_tokenizer(vlm_cfg.lm.tokenizer, vlm_cfg.image_token)
    image_processor = get_image_processor(vlm_cfg.vit.img_size)

    if train_cfg.dataset_type == 'flickr':
        print(f"Loading dataset from disk: {train_cfg.dataset_local_path}")
        raw = load_from_disk(train_cfg.dataset_local_path)
        ds = raw["train"] if "train" in raw else raw

        from data.dataset import FlickrDataset
        train_dataset = FlickrDataset(
            ds, tokenizer, image_processor, vlm_cfg
        )
        val_dataset = FlickrDataset(
            ds, tokenizer, image_processor, vlm_cfg
        )
    else:
        import hashlib        

        # Load and concatenate all cauldron subsets
        train_splits = []
        val_splits = []
        base_path = train_cfg.dataset_local_path
        for subset in train_cfg.dataset_subsets:
            subset_path = os.path.join(base_path, subset)
            if not os.path.exists(subset_path):
                print(f"  [skip] {subset} not found at {subset_path}")
                continue
            print(f"  Loading {subset}...")
            raw = load_from_disk(subset_path)
            ds = raw["train"] if "train" in raw else raw

            subset_hash = hashlib.md5(subset.encode()).hexdigest()[:8]
            train_cache = f"/tmpdir/user/hf_cache/{subset_hash}_train.arrow"
            test_cache  = f"/tmpdir/user/hf_cache/{subset_hash}_test.arrow"

            n_val = int(len(ds) * train_cfg.val_proportion)
            indices = list(range(len(ds)))
            train_splits.append(ds.select(indices[n_val:],  indices_cache_file_name=train_cache))
            val_splits.append(  ds.select(indices[:n_val],  indices_cache_file_name=test_cache))

        if not train_splits:
            raise ValueError(
                f"No cauldron subsets found under {base_path}/. "
                "Run prepare_datasets.py first."
            )

        train_ds = concatenate_datasets(train_splits)
        val_ds = concatenate_datasets(val_splits)
        print(f"Loaded {len(train_splits)} subsets → {len(train_ds)} train / {len(val_ds)} val samples")

        from data.dataset import CauldronDataset
        train_dataset = CauldronDataset(
            train_ds, tokenizer, image_processor, vlm_cfg
        )
        val_dataset = CauldronDataset(
            val_ds, tokenizer, image_processor, vlm_cfg
        )

    collator = VQACollator(tokenizer, max_length=train_cfg.max_length)

    # Sampler : shuffle and start again at the correct sample
    shuffle_seed = 42 # Do not change (jobs have to use the same permutation)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=collator,
        num_workers=1,
        pin_memory=True    
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=collator,
        num_workers=1,
        pin_memory=True,
    )
    return train_loader, val_loader


# ========================================
# checkpoints helpers (vibe coded)

import shutil

def _unwrap(model):
    # torch.compile wraps the model in OptimizedModule
    return model._orig_mod if hasattr(model, "_orig_mod") else model

def save_checkpoint(path, model, global_step, best_val_loss, best_mmstar_acc, optimizer=None):
    """Write atomically: build in a tmp dir, then swap in, so a job killed
    mid-save can never leave a corrupted checkpoint behind."""
    tmp = path + "_tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    _unwrap(model).save_pretrained(tmp)
    if optimizer is not None:
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "best_mmstar_acc": best_mmstar_acc,
            },
            os.path.join(tmp, "training_state.pt"),
        )
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.replace(tmp, path)  # atomic on the same filesystem

def find_resume_state(checkpoint_dir, device):
    last_dir = os.path.join(checkpoint_dir, "last")
    state_path = os.path.join(last_dir, "training_state.pt")
    if os.path.isdir(last_dir) and os.path.isfile(state_path):
        return last_dir, torch.load(state_path, map_location=device)
    return None, None

#=========================================




# ── Main training function ────────────────────────────────────────────────────
def train(train_cfg: TrainConfig, vlm_cfg: VLMConfig):
    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.backends.mps.enable_fallback_to_cpu = True
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    # ======================= Vibe coded checkpoints
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    resume_dir, resume_state = find_resume_state(train_cfg.checkpoint_dir, device)

    if resume_dir:
        print(f"Resuming from {resume_dir} (step {resume_state['global_step']})")
        model = VisionLanguageModel.from_pretrained(resume_dir)
    else:
        print("No checkpoint found — starting from scratch.")
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.load_backbone_weights)
    model.to(device)
    if train_cfg.compile:
        model = torch.compile(model)
    # ===================================================

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # ── Optimizer — three learning-rate groups ────────────────────────────────
    # Why three groups?
    #   * The modality projector (MP) is randomly initialised → high LR
    #   * The ViT and LM are pretrained → low LR to preserve knowledge
    param_groups = [
        {
            "params": list(model.MP.parameters()),
            "lr": train_cfg.lr_mp,
            "name": "MP",
        },
        {
            "params": list(model.vision_encoder.parameters()),
            "lr": train_cfg.lr_vit,
            "name": "ViT",
        },
        {
            "params": list(model.decoder.parameters()),
            "lr": train_cfg.lr_lm,
            "name": "LM",
        },
    ]
    # max_lrs: the initial (maximum) LR per group, used inside get_lr().
    # Students reference this in TODO 5.
    max_lrs = [  # noqa: F841
        train_cfg.lr_mp, train_cfg.lr_vit, train_cfg.lr_lm
    ]
    # =========================== Vibe code checkpoints
    optimizer = optim.AdamW(param_groups)
    all_params = [p for g in optimizer.param_groups for p in g["params"]]
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])

    global_step = resume_state["global_step"] if resume_state else 0
    best_val_loss = resume_state["best_val_loss"] if resume_state else float("inf")
    best_mmstar_acc = resume_state["best_mmstar_acc"] if resume_state else -1.0
    samples_consumed = resume_state["samples_consumed"] if resume_state else 0
    # ==================================================

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(train_cfg, vlm_cfg, samples_consumed)
    iter_train = iter(train_loader)

    # ── AMP context ───────────────────────────────────────────────────────────
    autocast_dtype = (
        torch.bfloat16 if device.type in ("cuda", "cpu") else torch.float16
    )
    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=autocast_dtype
    )

    # ── Checkpoint directory ──────────────────────────────────────────────────
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    # ── Training state ────────────────────────────────────────────────────────
    batch_loss = 0.0   # set by the student section each micro-step
    optimizer.zero_grad()

    print(
        f"Training for {train_cfg.max_steps} optimiser steps "
        f"(gradient_accumulation={train_cfg.gradient_accumulation_steps})"
    )
    t0 = time.time()

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    accum_step = 0   # counts micro-steps within one accumulation cycle
    while global_step < train_cfg.max_steps:
        model.train()

        # ── Get next batch (skip None batches from the collator) ──────────────
        batch = None
        while batch is None:
            try:
                batch = next(iter_train)
            except StopIteration:
                #iter_train = iter(train_loader)
                #batch = next(iter_train)
                # For practical reason, we will only do 1 epoch (no time remaining + simpler to handle dataloader logic)
                print(f"Epoch complete after {samples_consumed} samples — stopping.")
                break

        is_update_step = (
            (accum_step + 1) % train_cfg.gradient_accumulation_steps == 0
        )

        # ══════════════════════════════════════════════════════════════════════
        # STUDENT SECTION — implement the training step
        #
        # TODO 1 — Move all four batch tensors (input_ids, pixel_values,
        #           attention_mask, labels) to the training device.
        input_ids, pixel_values, attention_mask, labels = (
            batch['input_ids'],
            batch['pixel_values'],
            batch['attention_mask'],
            batch['labels'],
        )

        samples_consumed += input_ids.size(0) # keep track for restarting at the good sample

        """print(input_ids.shape, labels.shape)
        for i, element in enumerate(input_ids[0]):
            print(i, element, labels[0, i])
        # Set labels to random intergers to see if prediction are random
        labels = torch.randint_like(labels, 0, 4)""" # debugging tools

        input_ids = input_ids.to(device)
        pixel_values = pixel_values.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        #
        # TODO 2 — Run the forward pass inside autocast_ctx for mixed-precision
        #           training. The model returns (logits, loss); discard logits.
        with autocast_ctx:
            _, loss = model(input_ids, pixel_values, attention_mask=attention_mask, targets=labels)
        # TODO 3 — Divide the loss by gradient_accumulation_steps before
        #           backpropagating, so gradients accumulate correctly across
        #           micro-steps.
        loss /= train_cfg.gradient_accumulation_steps
        # TODO 4 — Call backward on the scaled loss to accumulate gradients.
        loss.backward()

        # TODO 5 — On update steps only (is_update_step):
        #           Clip gradients with torch.nn.utils.clip_grad_norm_ using
        #           all_params and max_grad_norm. Update each param group's
        #           "lr" by calling get_lr with global_step, the group's
        #           corresponding entry from max_lrs, and max_steps. Then step
        #           the optimizer, zero its gradients, and increment global_step.
        if is_update_step:
            torch.nn.utils.clip_grad_norm_(all_params, train_cfg.max_grad_norm)
            for i in range(len(param_groups)):
                lr = get_lr(global_step, max_lr=max_lrs[i], max_steps=train_cfg.max_steps)
                optimizer.param_groups[i]["lr"] = lr
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1


        #
        # TODO 6 — Record the unscaled loss in batch_loss for logging by
        #           reversing the accumulation scaling on the detached scalar
        #           (use .item() to detach from the graph).
        batch_loss = loss.item() * train_cfg.gradient_accumulation_steps

        # ══════════════════════════════════════════════════════════════════════

        accum_step += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if is_update_step and global_step % train_cfg.log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"step {global_step:5d} | loss {batch_loss:.4e}"
                f" | {elapsed:.1f}s"
            )

        # ── Evaluation ────────────────────────────────────────────────────────
        if is_update_step and global_step % train_cfg.eval_interval == 0:
            model.eval()
            val_losses = []
            val_iter = iter(val_loader)
            n_val = min(64, train_cfg.val_size // train_cfg.batch_size)
            for _ in range(n_val):
                vbatch = next(val_iter, None)
                if vbatch is None:
                    break
                with torch.no_grad(), autocast_ctx:
                    _, vloss = model(
                        vbatch["input_ids"].to(device),
                        vbatch["pixel_values"].to(device),
                        vbatch["attention_mask"].to(device),
                        vbatch["labels"].to(device),
                    )
                if vloss is not None:
                    val_losses.append(vloss.item())

            # ====================== checkpoints
            avg_val = sum(val_losses) / len(val_losses) if val_losses else float("nan")
            print(f"step {global_step:5d} | val_loss {avg_val:.4e}")

            is_new_best = avg_val < best_val_loss
            if is_new_best:
                best_val_loss = avg_val

            last_ckpt = os.path.join(train_cfg.checkpoint_dir, "last")
            save_checkpoint(last_ckpt, model, global_step, best_val_loss, best_mmstar_acc, optimizer=optimizer)
            print(f"  → saved last checkpoint to {last_ckpt}")

            if is_new_best:
                best_ckpt = os.path.join(train_cfg.checkpoint_dir, "best")
                save_checkpoint(best_ckpt, model, global_step, best_val_loss, best_mmstar_acc)
                print(f"  → new best checkpoint saved to {best_ckpt}")
            # =====================================

            # We don't use mmstar as validation because we use it in test
            if (
                train_cfg.mmstar_val_path
                and train_cfg.mmstar_eval_interval > 0
                and global_step % train_cfg.mmstar_eval_interval == 0
            ):
                from datasets import load_from_disk

                from eval_mmstar import evaluate_mmstar

                tokenizer = get_tokenizer(vlm_cfg.lm.tokenizer, vlm_cfg.image_token)
                image_processor = get_image_processor(vlm_cfg.vit.img_size)
                raw_mmstar = load_from_disk(train_cfg.mmstar_val_path)
                mmstar_val = raw_mmstar["val"] if "val" in raw_mmstar else raw_mmstar
                mmstar_metrics = evaluate_mmstar(
                    model=model,
                    dataset=mmstar_val,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    device=device,
                    limit=train_cfg.mmstar_eval_limit,
                    show_progress=False,
                )
                mmstar_acc = mmstar_metrics["accuracy"]
                print(
                    f"step {global_step:5d} | mmstar_val_acc "
                    f"{mmstar_acc:.4f}"
                )

                os.makedirs(train_cfg.mmstar_output_dir, exist_ok=True)
                mmstar_path = os.path.join(
                    train_cfg.mmstar_output_dir,
                    f"mmstar_step{global_step}.json",
                )
                with open(mmstar_path, "w") as f:
                    json.dump(
                        {
                            "global_step": global_step,
                            "checkpoint_dir": train_cfg.checkpoint_dir,
                            "mmstar_val_path": train_cfg.mmstar_val_path,
                            "metrics": mmstar_metrics,
                        },
                        f,
                        indent=2,
                    )

                if mmstar_acc > best_mmstar_acc:
                    # ==================== We don't want checkpoints from mmstar
                    best_mmstar_acc = mmstar_acc
                    print(f"  → new best MMStar accuracy: {best_mmstar_acc:.4f} (recorded, no separate checkpoint)")
                    # ===========================================================

            model.train()

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the VLM project model"
    )
    for f in fields(TrainConfig):
        if f.type in (int, float, bool, str):
            parser.add_argument(
                f"--{f.name}",
                type=f.type,
                default=None,
                help=f"TrainConfig.{f.name} (default: {f.default})",
            )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig()
    for f in fields(TrainConfig):
        val = getattr(args, f.name, None)
        if val is not None:
            setattr(train_cfg, f.name, val)
    train(train_cfg, vlm_cfg)
