# Project: Build a Vision-Language Model

In this project you will implement a full Vision-Language Model (VLM) from a skeleton codebase.
The goal is to produce a model that can accept an image and a text prompt, and generate a descriptive answer — exactly like small open-source models such as SmolVLM.

You will implement the two backbone encoders (a Vision Transformer and a Language Model), a lightweight modality projector that bridges them, and the training loop. A complete test suite lets you validate each component independently before wiring everything together.

**Groups.** The project is done in groups. Choose one person's GitHub repository as the shared workspace; that person must invite the other group members as collaborators so everyone can push and contribute.

**Infrastructure.** Training will be conducted on [Turpan](turpan_cheatsheet.md). Refer to the Turpan cheatsheet for how to connect, submit jobs, and manage the cluster.

---

## Section 1 — Overview

### Architecture

```
Image [B, 3, 512, 512]
    │
ViT (SigLIP2-base-patch16-512, ~86 M params)
    │  [B, 1024, 768]   — one 768-dim token per 16×16 patch
    │
Modality Projector  (pixel-shuffle ×4 → linear)
    │  [B, 64, 960]    — compressed to 64 tokens in LM space
    │
Language Model (SmolLM2-360M-Instruct, ~362 M params)
    │  [B, T, 960]
    │
Output head → logits [B, T, vocab_size]
```

The image is tokenised into **64 visual tokens** that are injected into the LM's token sequence in place of `<|image|>` placeholder tokens.

### Workflow

| Stage | What you implement | How you verify |
|---|---|---|
| 1. Backbones | `__init__` of ViT and LM components | [fast shape tests](#section-6-test-suite) |
| 2. Weight loading | `from_pretrained` succeeds | [slow pretrained tests](#section-6-test-suite) |
| 3. Forwards | `forward` of every component | [shape + correctness tests](#section-6-test-suite) |
| 4. Glue | `VisionLanguageModel` | [end-to-end VLM tests](#section-6-test-suite) |
| 5. Training | inner loop in `train.py` | smoke-test on Flickr |
| 6. Evaluation | TBD | — |

### Models and key numbers

**ViT — `google/siglip2-base-patch16-512`**

| Parameter | Value |
|---|---|
| `img_size` | 512 |
| `patch_size` | 16 |
| `num_patches` | (512 / 16)² = **1024** |
| `hidden_dim` | 768 |
| `inter_dim` | 3072 |
| `n_heads` | 12 (head\_dim = 64) |
| `n_blocks` | 12 |
| Total params | ~86 M |

**Language Model — `HuggingFaceTB/SmolLM2-360M-Instruct`**

| Parameter | Value |
|---|---|
| `hidden_dim` | 960 |
| `inter_dim` | 2560 |
| `n_heads` | 15 (head\_dim = 64) |
| `n_kv_heads` | 5 (GQA — 3 Q heads per KV head) |
| `n_blocks` | 32 |
| `vocab_size` | 49 153 (49 152 + 1 image token) |
| Weight tying | embedding ↔ output head |
| Total params | ~362 M |

**Modality Projector**

| Parameter | Value |
|---|---|
| `pixel_shuffle_factor` | 4 |
| Input | [B, 1024, 768] |
| After pixel-shuffle | [B, 64, 12 288] |
| After linear | [B, **64**, **960**] |

---

## Section 2 — ViT and Language Model

### Step 1 — Fill in the `__init__` methods

Open `models/vision_transformer.py` and `models/language_model.py`. Each class has a skeleton `__init__` with `TODO` comments. Fill them one class at a time, then immediately run the [corresponding test](#section-6-test-suite).

**ViT components** (implement in order):

| Class | Key constraint |
|---|---|
| `ViTPatchEmbeddings` | `conv` must have `kernel_size == stride == patch_size`; `position_embedding` is `[1, num_patches, hidden_dim]` |
| `ViTAttention` | fuse Q/K/V into a single `qkv_proj` (required for weight loading); both projections need `bias=True` |
| `ViTMLP` | `fc1`, `fc2`; activation must be **GELU** (not ReLU, not SiLU) |
| `ViTBlock` | name your attributes exactly `ln1`, `ln2`, `attn`, `mlp` |
| `ViT` | `patch_embedding`, `blocks` (a `nn.ModuleList`), `layer_norm` |

**LM components** (implement in order):

| Class | Key constraint |
|---|---|
| `RMSNorm` | no bias — only a gain weight; store `rms_eps` |
| `RotaryEmbedding` | precompute `inv_freq` of shape `[head_dim / 2]` — provided as a reference |
| `LMAttention` | separate `q_proj`, `k_proj`, `v_proj`, `out_proj`; k/v use `n_kv_heads × head_dim`; no biases; store `n_kv_groups` |
| `LMMLP` | SwiGLU: `gate_proj`, `up_proj`, `down_proj`; no biases |
| `LMBlock` | name your attributes exactly `norm1`, `norm2`, `attn`, `mlp` |
| `LanguageModel` | `token_embedding`, `blocks`, `head`; tie `head.weight = token_embedding.weight` |

!!! warning "Attribute names matter"
    `from_pretrained` maps HuggingFace checkpoint keys to your attribute names.
    Any mismatch will raise a `KeyError` when loading weights.
    Follow the names in the tables above exactly.

### Step 2 — Verify weight loading

Once all `__init__` methods pass their [fast tests](#section-6-test-suite), check that pretrained weights load cleanly.
These [tests](#section-6-test-suite) download ~800 MB from HuggingFace and require a GPU or a patient CPU:

```bash
pytest tests/test_pretrained_loading.py -m slow
```

If `from_pretrained` raises a `KeyError`, the attribute name in your `__init__` does not match the checkpoint. Compare the error key with the names listed in the tables above.

### Step 3 — Implement the `forward` methods

After the `__init__` [tests](#section-6-test-suite) pass, implement `forward` for each component and run the [tests](#section-6-test-suite) again. The test docstrings explain exactly what each forward must do.

---

## Section 3 — Modality Projector (Glue)

Open `models/modality_projector.py`. The projector bridges the ViT output (768-dim, 1024 tokens) and the LM input (960-dim, 64 tokens).

You have full freedom in how you implement it, as long as the interface is respected:

- `forward(x)` takes ViT features `[B, 1024, 768]` and returns `[B, 64, 960]`.
- The reference design uses **pixel-shuffle** followed by a **bias-free linear** projection.

**Pixel-shuffle** is a spatial downsampling technique borrowed from super-resolution. With `factor=4`, each group of 4×4 neighbouring patches is merged into a single wider token:

```
1024 tokens × 768 dims  →  64 tokens × 12 288 dims  →  64 tokens × 960 dims
         (spatial compression)                    (dimension projection)
```

The [`ModalityProjector` tests](#section-6-test-suite) verify the shapes at each stage for a default config, but **you can change it later on since you will train it from scratch**:

```bash
pytest tests/test_modality_projector.py
```

---

## Section 4 — Training

Open `train.py`. The script is fully provided except for the inner training loop body, which contains six clearly labelled `TODO` comments:

| TODO | What to implement |
|---|---|
| 1 | Move batch tensors to the correct device |
| 2 | Forward pass inside `autocast_ctx` for mixed precision |
| 3 | Scale the loss for gradient accumulation |
| 4 | Backward pass |
| 5 | Optimiser step with gradient clipping and LR scheduling (update steps only) |
| 6 | Store the unscaled loss for logging |

Everything else — the optimiser, three per-component learning-rate groups, the cosine+warmup schedule, evaluation loop, and checkpoint saving — is already implemented.

### Datasets

Prepare datasets before training using `prepare_datasets.py`:

```bash
# Flickr30k — lightweight, good for a smoke test
python prepare_datasets.py --dataset flickr --output_dir /path/to/flickr30k

# The Cauldron — large multi-task dataset for serious training
python prepare_datasets.py --dataset cauldron --output_dir /path/to/cauldron
```

Then launch training:

```bash
# Quick smoke test on Flickr
python train.py --dataset_type flickr \
    --dataset_local_path /path/to/flickr30k \
    --max_steps 100 --batch_size 1

# Full training on The Cauldron
python train.py --dataset_type cauldron \
    --dataset_local_path /path/to/cauldron \
    --max_steps 10000
```

### Recommended learning rates

| Component | Default LR | Rationale |
|---|---|---|
| Modality Projector | `5e-3` | randomly initialised — needs a strong signal |
| ViT | `5e-5` | pretrained — fine-tune gently |
| LM | `5e-5` | pretrained — fine-tune gently |

---

## Section 5 — Evaluation

> **TBD** — evaluation metrics and scripts will be provided in a later update.

---

## Section 6 — Test Suite

The project ships with five test files. Four run entirely on CPU without any downloads; one downloads pretrained weights and is skipped by default.

### Running the tests

```bash
# All fast tests (no downloads, runs in seconds)
pytest tests/

# A single test file
pytest tests/test_vision_transformer.py

# A single test class
pytest tests/test_language_model.py::TestRMSNorm

# A single test
pytest tests/test_language_model.py::TestLanguageModel::test_kv_cache_consistency

# Slow tests (downloads ~800 MB, requires GPU for reasonable speed)
pytest -m slow
```

### Test files

#### `test_vision_transformer.py` — fast

Tests every ViT building block in isolation with a tiny config (`img_size=64`, `hidden_dim=32`, 2 blocks) so each test runs in milliseconds.

| Class | What it checks |
|---|---|
| `TestViTPatchEmbeddings` | conv shape and stride, position embedding shape, position embedding actually added |
| `TestViTAttention` | fused `qkv_proj` shape, output shape, bidirectional (no causal mask) |
| `TestViTMLP` | `fc1`/`fc2` shapes, GELU activation |
| `TestViTBlock` | attribute names, residual connection |
| `TestViT` | end-to-end shape, dtype, arbitrary batch sizes |

Also contains `TestViTPretrainedLoading` (marked `slow`) that loads real SigLIP2 weights and checks parameter count and config values.

#### `test_language_model.py` — fast

Tests every LM building block with a tiny config (`hidden_dim=32`, `n_heads=4`, `n_kv_heads=2`, 2 blocks).

| Class | What it checks |
|---|---|
| `TestRMSNorm` | weight shape, scale invariance, weight effect |
| `TestRotaryEmbedding` | `inv_freq` shape |
| `TestLMAttention` | GQA projection shapes, no biases, KV cache shape |
| `TestLMMlp` | SwiGLU projection shapes |
| `TestLMBlock` | attribute names, output shape |
| `TestLanguageModel` | weight tying, forward shape, KV cache consistency |

The **`test_kv_cache_consistency`** test is the key correctness check: it verifies that running the full sequence at once (prefill) produces the same logits as running a prefix to build the KV cache and then decoding one token at a time. A failure here points to a bug in your KV cache concatenation or RoPE `start_pos` handling.

Also contains `TestLanguageModelPretrainedLoading` (marked `slow`) that loads real SmolLM2-360M weights.

#### `test_modality_projector.py` — fast

Tests the projector against the real ViT/LM hidden dimensions (768 and 960).

| Test | What it checks |
|---|---|
| `test_init` | `input_dim` computation, `proj` is bias-free `nn.Linear` |
| `test_pixel_shuffle_shape` | 1024 × 768 → 64 × 12 288 |
| `test_forward_shape` | full pipeline 1024 × 768 → 64 × 960 |
| `test_dtype_preserved` | no silent float16 cast |

#### `test_vlm.py` — fast

End-to-end tests for `VisionLanguageModel` using tiny config and `load_backbone=False` (no HuggingFace downloads).

| Test | What it checks |
|---|---|
| `test_forward_without_targets` | returns `(hidden_states, None)`, shape `[B, T, hidden_dim]` |
| `test_forward_with_targets` | returns `(logits, scalar_loss)`, loss is finite |
| `test_image_token_replacement` | image token positions are replaced by projected ViT features |

!!! tip "Check for NaN loss"
    If `test_forward_with_targets` fails with a non-finite loss, the most likely cause is incorrect cross-entropy masking. Image token positions must use `ignore_index=-100` in the targets so they do not contribute to the loss.

#### `test_pretrained_loading.py` — slow

Integration tests that download real weights and verify the full pipeline end-to-end.

| Class | What it checks |
|---|---|
| `TestViTPretrainedLoading` | weight loading, ~86 M param count, forward shape `[1, 1024, 768]` |
| `TestLMPretrainedLoading` | weight loading, ~362 M param count, forward shape, KV cache consistency with real weights |
| `TestVLMPretrainedForward` | full VLM forward with pretrained ViT + LM, finite loss |

Run with:

```bash
pytest tests/test_pretrained_loading.py -m slow
```

### Recommended test order

1. `pytest tests/test_vision_transformer.py` — fix all ViT `__init__` and `forward` issues
2. `pytest tests/test_language_model.py` — fix all LM issues, especially KV cache
3. `pytest tests/test_modality_projector.py` — fix the projector
4. `pytest tests/test_vlm.py` — fix the full VLM wiring
5. `pytest -m slow` — confirm pretrained weight loading works before training
