from __future__ import annotations

from taproot.util import PretrainedModelMixin, get_added_token_dict

from typing import Optional, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import CLIPTokenizer # type: ignore[import-untyped]

__all__ = [
    "CLIPViTLTokenizer"
]

class CLIPViTLTokenizer(PretrainedModelMixin):
    """
    Tokenizer for OpenAI's CLIP (Contrastive Language-Image Pretraining) model with a ViT-Large backbone.
    """
    init_file_urls = {
        "vocab_file": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-clip-vit-l-tokenizer-vocab.json",
        "special_tokens_map_file": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-clip-vit-l-tokenizer-special-tokens-map.json",
        "merges_file": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-clip-vit-l-tokenizer-merges.txt",
    }

    @classmethod
    def get_model_class(cls) -> Type[CLIPTokenizer]:
        """
        Get the model class for the FLUX Tokenizer.
        """
        from transformers import CLIPTokenizer
        return CLIPTokenizer # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        et the default configuration for the FLUX Tokenizer.
        """
        return {
            "add_prefix_space": False,
            "added_tokens_decoder": get_added_token_dict({
                "49406": {
                    "content": "<|startoftext|>",
                    "lstrip": False,
                    "normalized": True,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "49407": {
                    "content": "<|endoftext|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                }
            }),
            "bos_token": "<|startoftext|>",
            "clean_up_tokenization_spaces": True,
            "do_lower_case": True,
            "eos_token": "<|endoftext|>",
            "errors": "replace",
            "model_max_length": 77,
            "pad_token": "<|endoftext|>",
            "tokenizer_class": "CLIPTokenizer",
            "unk_token": "<|endoftext|>"
        }
