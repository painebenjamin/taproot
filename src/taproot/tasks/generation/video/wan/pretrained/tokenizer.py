from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union, TYPE_CHECKING

from taproot.util import PretrainedModelMixin, get_added_token_dict

if TYPE_CHECKING:
    from transformers import T5Tokenizer # type: ignore[import-untyped]

__all__ = ["PretrainedWanTokenizer"]

class PretrainedWanTokenizer(PretrainedModelMixin):
    """
    The Wan tokenizer class (multilingual T5)
    """
    init_file_urls = {
        "vocab_file": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-umt5-xxl-vocab.model",
        "special_tokens_map_file": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-umt5-xxl-special-tokens-map.json",
    }

    @classmethod
    def get_model_class(cls) -> Type[T5Tokenizer]:
        """
        Returns the model class.
        """
        from transformers import T5Tokenizer
        return T5Tokenizer # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
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
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "3": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
        }

        for i in range(300):
            added_tokens_decoder[f"{256000 + i}"] = {
                "content": f"<extra_id_{299-i}>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }

        return {
            "added_tokens_decoder": get_added_token_dict(added_tokens_decoder),
            "additional_special_tokens": [f"<extra_id_{i}>" for i in range(300)],
            "bos_token": "<s>",
            "clean_up_tokenization_spaces": True,
            "eos_token": "</s>",
            "extra_ids": 300,
            "model_max_length": 1000000000000000019884624838656,
            "pad_token": "<pad>",
            "sp_model_kwargs": {},
            "spaces_between_special_tokens": False,
            "tokenizer_class": "T5Tokenizer",
            "legacy": True,
            "unk_token": "<unk>"
        }
