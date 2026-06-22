from dataclasses import dataclass, field
import os
from typing import Tuple


@dataclass
class ViTConfig:
    """Configuration for the Vision Transformer (SigLIP2-base-patch16-512).

    Passed directly to ViT and its sub-modules, so attribute names match
    the model's own attribute names (hidden_dim, n_heads, …).
    """
    # Input:  [B, 3, 512, 512]  →  (512/16)² = 1024 patches per image
    # Output: [B, 1024, 768]

    # Embedding dimension of each patch token; used as in/out size of ViTAttention,
    # ViTMLP, LayerNorm, and the Conv2d out_channels in ViTPatchEmbeddings
    hidden_dim: int = 768

    # Inner (expanded) dimension of the 2-layer ViTMLP (fc1 output / fc2 input)
    inter_dim: int = 3072

    # Kernel size AND stride of the Conv2d in ViTPatchEmbeddings;
    # determines patch grid resolution: (img_size/patch_size)² = num_patches
    patch_size: int = 16

    # Expected spatial resolution of input images; also used by get_image_processor()
    # to build the Resize transform, and to compute num_patches in ViTPatchEmbeddings
    img_size: int = 512

    # Number of attention heads in ViTAttention (head_dim = hidden_dim / n_heads = 64);
    # controls the reshape [B, T, C] → [B, n_heads, T, head_dim] inside forward
    n_heads: int = 12

    # Number of sequential ViTBlock layers stacked in the encoder
    n_blocks: int = 12

    # Dropout probability applied in ViTAttention (attn weights + residual)
    # and in ViTMLP; set to 0 for inference / pretrained models
    dropout: float = 0.0

    # Epsilon for LayerNorm in each ViTBlock (ln1, ln2)
    ln_eps: float = 1e-6

    # Whether to use a CLS token; False means all 1024 patch tokens are
    # forwarded to the modality projector (no cls pooling)
    cls_flag: bool = False

    # HuggingFace model identifier; used by ViT.from_pretrained() to download
    # the pretrained SigLIP2 weights from the Hub
    model_type: str = 'google/siglip2-base-patch16-512'


@dataclass
class LMConfig:
    """Configuration for the Language Model (SmolLM2-360M-Instruct).

    Passed directly to LanguageModel and its sub-modules.
    """
    # Input:  [B, T, 960]  →  Output: [B, T, 960]

    # Embedding dimension of each token; sizes the Embedding table, RMSNorm,
    # LMAttention projections (q_proj, out_proj), LMMLP, and the output head
    hidden_dim: int = 960

    # Inner dimension of the gated MLP (gate_proj and up_proj output / down_proj input)
    inter_dim: int = 2560

    # Epsilon for RMSNorm (used in every LMBlock norm1/norm2 and final norm)
    rms_eps: float = 1e-5

    # Base frequency for Rotary Position Embeddings (RoPE);
    # used in RotaryEmbedding to compute inv_freq = 1/(re_base^(2i/dim))
    re_base: int = 100000

    # Maximum sequence length supported by RoPE; if exceeded, frequencies
    # are scaled down proportionally inside RotaryEmbedding.forward()
    max_position_embeddings: int = 8192

    # Original SmolLM2 vocabulary size before adding the <|image|> token
    base_vocab_size: int = 49152

    # Actual vocabulary size (base + 1 image token); sizes the Embedding
    # table (token_embedding) and the output Linear head
    vocab_size: int = 49153

    # Number of query heads in LMAttention; determines q_proj output size
    # (n_heads × head_dim) and the Q reshape in the GQA forward pass
    n_heads: int = 15

    # Number of key/value heads (Grouped-Query Attention); k_proj and v_proj
    # output n_kv_heads × head_dim; each KV head is shared by n_heads/n_kv_heads=3 Q heads
    n_kv_heads: int = 5

    # Number of sequential LMBlock layers in the decoder stack
    n_blocks: int = 32

    # Dropout probability in LMAttention (attn weights + residual);
    # 0 at inference / for pretrained weights
    dropout: float = 0.0

    # Multiplicative scaling applied to cos/sin embeddings in RotaryEmbedding;
    # 1.0 means no extra scaling beyond the standard RoPE formulation
    attn_scaling: float = 1.0

    # If True, the output head (Linear: hidden_dim → vocab_size) shares its
    # weight matrix with token_embedding.weight (saves parameters)
    tie_weights: bool = True

    # HuggingFace model identifier; used by LanguageModel.from_pretrained()
    # to download pretrained SmolLM2-360M weights from the Hub
    model_type: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'

    # Tokenizer identifier; passed to get_tokenizer() which loads the
    # AutoTokenizer and adds the <|image|> special token to its vocabulary
    tokenizer: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'


@dataclass
class ProjectorConfig:
    """Configuration for the Modality Projector.

    Passed to ModalityProjector alongside ViTConfig and LMConfig.
    """

    # Spatial downsampling factor for pixel_shuffle(); groups factor² neighbouring
    # ViT patches into one token, increasing its embedding dim by factor²
    # (768 × 16 = 12288) while reducing token count (1024 → 64)
    pixel_shuffle_factor: int = 4

    # Number of visual tokens output by the projector (= num_patches / factor²);
    # also the number of <|image|> placeholders inserted into the prompt
    image_token_length: int = 64


@dataclass
class VLMConfig:
    """Full VLM configuration (groups the three sub-configs)."""
    vit: ViTConfig = field(default_factory=ViTConfig)
    lm: LMConfig = field(default_factory=LMConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)

    # Special token string added to the tokenizer as an additional_special_token;
    # repeated image_token_length=64 times in the prompt to mark positions that
    # _replace_img_tokens_with_embd() will overwrite with projected visual embeddings
    image_token: str = '<|image|>'

    # Whether to download and load pretrained SigLIP2 + SmolLM2 weights
    # when constructing VisionLanguageModel (False for unit tests)
    load_backbone_weights: bool = True

    # Directory where VisionLanguageModel.save_pretrained() writes the
    # safetensors checkpoint and config JSON during training
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

    # AdamW learning rate for the Modality Projector parameter group;
    # high because MP is randomly initialised (no pretrained weights)
    lr_mp: float = 5e-3

    # AdamW learning rate for the ViT (vision encoder) parameter group;
    # low to preserve pretrained SigLIP2 features
    lr_vit: float = 5e-5

    # AdamW learning rate for the Language Model (decoder) parameter group;
    # low to preserve pretrained SmolLM2 knowledge
    lr_lm: float = 5e-5

    # Number of samples per forward pass (micro-batch); limited by GPU memory
    batch_size: int = 16

    # Number of micro-batches accumulated before one optimizer.step();
    # effective batch size = batch_size × gradient_accumulation_steps = 16
    gradient_accumulation_steps: int = 8

    # Maximum L2 norm for gradient clipping (torch.nn.utils.clip_grad_norm_);
    # applied to all parameters before each optimizer step
    max_grad_norm: float = 1.0

    # Total number of optimizer steps (not micro-steps); training loop ends
    # when global_step reaches this value
    max_steps: int = 10000

    # Run the validation loop every eval_interval optimizer steps;
    # computes average val loss and saves checkpoint if improved
    #eval_interval: int = 500
    eval_interval: int = 50

    # Print training loss to stdout every log_interval optimizer steps
    #log_interval: int = 50
    log_interval: int = 250

    # Fraction of max_steps used for linear LR warmup (0 → max_lr);
    # after warmup, LR decays via cosine schedule to max_lr/10
    warmup_fraction: float = 0.03

    # ─── Dataset ──────────────────────────────────────────────────────────────

    # Which dataset class to instantiate: 'cauldron' → CauldronDataset,
    # 'flickr' → FlickrDataset (both are IterableDatasets)
    dataset_type: str = 'cauldron'

    # Cauldron subset(s) to train on; only used when dataset_type == 'cauldron'.
    # Defaults to all available subsets.
    dataset_subsets: Tuple[str, ...] = (
        "ai2d", "aokvqa", "chart2text", "chartqa",
        "cocoqa", "datikz", "diagram_image_to_text", "docvqa", "dvqa",
        "figureqa", "finqa", "geomverse", "hateful_memes", "hitab", "iam",
        "iconqa", "infographic_vqa", "intergps", "localized_narratives",
        "mapqa", "mimic_cgd", "multihiertt", "nlvr2", "ocrvqa",
        "plotqa", "raven", "rendered_text", "robut_sqa", "robut_wikisql",
        "robut_wtq", "scienceqa", "screen2words", "spot_the_diff", "st_vqa",
        "tabmwp", "tallyqa", "tat_qa", "textcaps", "textvqa", "tqa",
        "vistext", "visual7w", "visualmrc", "vqarad", "vqav2", "vsr",
        "websight",
    )

    # Base path where prepare_datasets.py saved the Arrow datasets
    dataset_local_path: str = '/work/shared/TPIRT'

    # Number of validation samples evaluated at each eval_interval
    # (capped at 64 batches inside the eval loop)
    val_size: int = 256

    # Proportion of the dataset to use for validation
    val_proportion: float = 0.2

    # Maximum sequence length in tokens; samples longer than this are
    # dropped by VQACollator to prevent OOM
    max_length: int = 2048

    # Optional MMStar validation callback. Leave mmstar_val_path empty to disable.
    mmstar_val_path: str = '/work/shared/TPIRT/mmstar'
    mmstar_eval_interval: int = 0
    mmstar_eval_limit: int = 128
    mmstar_output_dir: str = 'eval_results'

    checkpoint_dir: str = '/work/formation/tpirtchrbd/checkpoints/best'

    # Whether to apply torch.compile() to the model for potential speedup
    compile: bool = False

    def __post_init__(self):
        # Auto-append dataset type/subset to base path if not already present
        if self.dataset_type == 'flickr' and not self.dataset_local_path.endswith('flickr30k'):
            self.dataset_local_path = os.path.join(self.dataset_local_path, 'flickr30k')
        elif self.dataset_type == 'cauldron' and 'the_cauldron' not in self.dataset_local_path:
            self.dataset_local_path = os.path.join(self.dataset_local_path, 'the_cauldron')