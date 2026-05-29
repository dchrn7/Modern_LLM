"""Image preprocessing and tokenizer utilities. PROVIDED — do not modify."""

from transformers import AutoTokenizer
import torchvision.transforms as transforms

_TOKENIZER_CACHE = {}


def get_tokenizer(model_name: str, image_token: str = '<|image|>'):
    """Load and cache the SmolLM2 tokenizer, adding the image placeholder token.

    The image_token is added as a special token so the tokenizer maps it to a
    single id (not split into sub-words).  Its id is stored as
    tokenizer.image_token_id for use in _replace_img_tokens_with_embd.
    """
    if model_name not in _TOKENIZER_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        if image_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'additional_special_tokens': [image_token]})
        tokenizer.image_token = image_token
        tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        _TOKENIZER_CACHE[model_name] = tokenizer
    return _TOKENIZER_CACHE[model_name]


def get_image_processor(img_size: int = 512):
    """Return a torchvision transform that resizes and normalises images.

    SigLIP2 was pretrained with mean=0.5, std=0.5 normalisation on 512×512
    images.  Using the same preprocessing ensures the pretrained features
    remain valid.

    Note: We use a plain resize (no tiling) which is sufficient for most
    Cauldron / Flickr tasks.  For OCR-heavy tasks (docvqa, ocrvqa), tiling
    would help but is out of scope for this project.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_image_string(mp_image_token_length: int, image_token: str = '<|image|>') -> str:
    """Return the image placeholder string to insert at the start of each prompt.

    The string contains mp_image_token_length (=64) copies of the image token.
    These positions are later replaced by the modality projector's output.
    """
    return image_token * mp_image_token_length
