from __future__ import annotations

from taproot.util import PretrainedModelMixin, get_added_token_dict

from typing import Optional, Dict, Any, Type, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import T5Tokenizer # type: ignore[import-untyped]

__all__ = ["T5XXLTokenizer"]

class T5XXLTokenizer(PretrainedModelMixin):
    """
    The secondary model for the FLUX Tokenizer.
    """
    init_file_urls = {
        "vocab_file": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-t5-xxl-vocab.model",
        "special_tokens_map_file": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-t5-xxl-special-tokens-map.json",
    }

    @classmethod
    def get_model_class(cls) -> Type[T5Tokenizer]:
        """
        Get the model class for the FLUX Tokenizer.
        """
        from transformers import T5Tokenizer
        return T5Tokenizer # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the FLUX Tokenizer.
        """
        added_tokens_decoder: Dict[Union[str, int], Dict[str, Union[str, bool]]] = {
            "0": {
                "content": "<pad>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
        }

        for i in range(100):
            added_tokens_decoder[f"{32000 + i}"] = {
                "content": f"<extra_id_{99-i}>",
                "lstrip": True,
                "normalized": False,
                "rstrip": True,
                "single_word": False,
                "special": True
            }

        return {
            "add_prefix_space": True,
            "added_tokens_decoder": get_added_token_dict(added_tokens_decoder),
            "additional_special_tokens": [f"<extra_id_{i}>" for i in range(100)],
            "clean_up_tokenization_spaces": True,
            "eos_token": "</s>",
            "extra_ids": 100,
            "legacy": True,
            "model_max_length": 512,
            "pad_token": "<pad>",
            "sp_model_kwargs": {},
            "tokenizer_class": "T5Tokenizer",
            "unk_token": "<unk>"
        }
