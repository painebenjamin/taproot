from __future__ import annotations

from taproot.util import PretrainedModelMixin

from typing import Optional, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import ( # type: ignore[import-untyped]
        CLIPTextConfig,
        CLIPTextModelWithProjection
    )

__all__ = [
    "OpenCLIPViTGTextEncoder"
]

class OpenCLIPViTGTextEncoder(PretrainedModelMixin):
    """
    OpenCLIP ViT-G Text Encoder.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-open-clip-vit-g.fp16.safetensors"
    use_strict = False

    @classmethod
    def get_model_class(cls) -> Type[CLIPTextModelWithProjection]:
        """
        Get the model class for the Stable Diffusion XL Text Encoder.
        """
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection # type: ignore[no-any-return]

    @classmethod
    def get_config_class(cls) -> Type[CLIPTextConfig]:
        """
        Get the configuration class for the Stable Diffusion XL Text Encoder.
        """
        from transformers import CLIPTextConfig
        return CLIPTextConfig # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Stable Diffusion XL Text Encoder.
        """
        return {
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "dropout": 0.0,
            "eos_token_id": 2,
            "hidden_act": "gelu",
            "hidden_size": 1280,
            "initializer_factor": 1.0,
            "initializer_range": 0.02,
            "intermediate_size": 5120,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 77,
            "model_type": "clip_text_model",
            "num_attention_heads": 20,
            "num_hidden_layers": 32,
            "pad_token_id": 1,
            "projection_dim": 1280,
            "torch_dtype": "float16",
            "vocab_size": 49408
        }
