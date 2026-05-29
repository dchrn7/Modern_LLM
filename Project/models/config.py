from dataclasses import dataclass, field


@dataclass
class ViTConfig:
    """Configuration for the Vision Transformer (SigLIP2-base-patch16-512).

    Passed directly to ViT and its sub-modules, so attribute names match
    the model's own attribute names (hidden_dim, n_heads, …).
    """
    # Input:  [B, 3, 512, 512]  →  (512/16)² = 1024 patches per image
    # Output: [B, 1024, 768]
    hidden_dim: int = 768
    inter_dim: int = 3072       # 4 × hidden_dim
    patch_size: int = 16
    img_size: int = 512
    n_heads: int = 12           # head_dim = 768 / 12 = 64
    n_blocks: int = 12
    dropout: float = 0.0
    ln_eps: float = 1e-6
    cls_flag: bool = False      # use all patch tokens, not a CLS token
    model_type: str = 'google/siglip2-base-patch16-512'


@dataclass
class LMConfig:
    """Configuration for the Language Model (SmolLM2-360M-Instruct).

    Passed directly to LanguageModel and its sub-modules.
    """
    # Input:  [B, T, 960]  →  Output: [B, T, 960]
    hidden_dim: int = 960
    inter_dim: int = 2560
    rms_eps: float = 1e-5
    re_base: int = 100000       # RoPE base frequency
    max_position_embeddings: int = 8192
    base_vocab_size: int = 49152
    vocab_size: int = 49153     # base + 1 image token <|image|>
    n_heads: int = 15           # query heads;  head_dim = 960 / 15 = 64
    n_kv_heads: int = 5         # key-value heads (GQA: 3 Q heads per KV head)
    n_blocks: int = 32
    dropout: float = 0.0
    attn_scaling: float = 1.0
    tie_weights: bool = True    # share input embeddings ↔ output head weights
    model_type: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    tokenizer: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'


@dataclass
class ProjectorConfig:
    """Configuration for the Modality Projector.

    Passed to ModalityProjector alongside ViTConfig and LMConfig.
    """
    pixel_shuffle_factor: int = 4
    image_token_length: int = 64    # 1024 / (4²) = 64


@dataclass
class VLMConfig:
    """Full VLM configuration (groups the three sub-configs)."""
    vit: ViTConfig = field(default_factory=ViTConfig)
    lm: LMConfig = field(default_factory=LMConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)

    # ─── Special tokens ───────────────────────────────────────────────────────
    # One extra token is added to the tokenizer vocabulary.
    # It is inserted projector.image_token_length=64 times in the prompt as
    # placeholders, then replaced by the modality projector's output embeddings.
    image_token: str = '<|image|>'

    load_backbone_weights: bool = True
    checkpoint_path: str = 'checkpoints'

    @classmethod
    def from_dict(cls, d: dict) -> 'VLMConfig':
        """Reconstruct from a plain dict (e.g. loaded from JSON via asdict)."""
        return cls(
            vit=ViTConfig(**d.get('vit', {})),
            lm=LMConfig(**d.get('lm', {})),
            projector=ProjectorConfig(**d.get('projector', {})),
            image_token=d.get('image_token', '<|image|>'),
            load_backbone_weights=d.get('load_backbone_weights', True),
            checkpoint_path=d.get('checkpoint_path', 'checkpoints'),
        )


@dataclass
class TrainConfig:
    # Learning rates — MP is randomly initialised (high LR); backbones are pretrained (low LR)
    lr_mp: float = 5e-3
    lr_vit: float = 5e-5
    lr_lm: float = 5e-5

    batch_size: int = 2
    gradient_accumulation_steps: int = 8   # effective batch = batch_size × grad_accum
    max_grad_norm: float = 1.0

    max_steps: int = 10000
    eval_interval: int = 500
    log_interval: int = 50
    warmup_fraction: float = 0.03          # 3% of max_steps used for LR warmup

    # ─── Dataset ──────────────────────────────────────────────────────────────
    # Datasets must be pre-downloaded with prepare_datasets.py (save_to_disk).
    # Set dataset_local_path to the directory produced by save_to_disk().
    dataset_type: str = 'cauldron'         # 'cauldron' or 'flickr'
    dataset_local_path: str = ''           # path to save_to_disk output (required)
    val_size: int = 256
    max_length: int = 2048

    checkpoint_dir: str = 'checkpoints'
    compile: bool = False
