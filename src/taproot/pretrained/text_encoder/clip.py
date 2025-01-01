from __future__ import annotations

from taproot.util import PretrainedModelMixin

from typing import Optional, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import ( # type: ignore[import-untyped]
        CLIPTextModel,
        CLIPTextConfig,
        CLIPTextModelWithProjection
    )

__all__ = [
    "CLIPViTLTextEncoder",
    "CLIPViTLTextEncoderWithProjection"
]

class CLIPViTLTextEncoder(PretrainedModelMixin):
    """
    The primary model for the CLIP ViT-L Text Encoder.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-clip-vit-l.bf16.safetensors"
    use_strict = False

    @classmethod
    def get_model_class(cls) -> Type[CLIPTextModel]:
        """
        Get the model class for the CLIP ViT-L Text Encoder.
        """
        from transformers import CLIPTextModel
        return CLIPTextModel # type: ignore[no-any-return]

    @classmethod
    def get_config_class(cls) -> Type[CLIPTextConfig]:
        """
        Get the configuration class for the CLIP ViT-L Text Encoder.
        """
        from transformers import CLIPTextConfig
        return CLIPTextConfig # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the CLIP ViT-L Text Encoder.
        """
        return {
            "attention_dropout": 0,
            "bos_token_id": 0,
            "dropout": 0,
            "eos_token_id": 2,
            "hidden_act": "quick_gelu",
            "hidden_size": 768,
            "initializer_factor": 1,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 0.00001,
            "max_position_embeddings": 77,
            "model_type": "clip_text_model",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 1,
            "projection_dim": 768,
            "torch_dtype": "bfloat16",
            "vocab_size": 49408
        }

class CLIPViTLTextEncoderWithProjection(CLIPViTLTextEncoder):
    """
    The model for the CLIP ViT-L Text Encoder with a projection head.
    """

    @classmethod
    def get_model_class(cls) -> Type[CLIPTextModelWithProjection]:
        """
        Get the model class for the CLIP ViT-L Text Encoder with a projection head.
        """
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection # type: ignore[no-any-return]
