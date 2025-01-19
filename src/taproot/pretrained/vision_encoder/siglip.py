from __future__ import annotations

from taproot.util import PretrainedModelMixin

from typing import Optional, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.models.siglip.modeling_siglip import ( # type: ignore[import-untyped]
        SiglipVisionConfig,
        SiglipVisionModel
    )

__all__ = [
    "SigLIPSO400MVisionEncoder",
]

class SigLIPSO400MVisionEncoder(PretrainedModelMixin):
    """
    SigLIP SO (shape-optimized) 400M Vision Encoder.
    Based on https://arxiv.org/abs/2305.13035 "Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design".
    """
    model_url = "https://huggingface.co/google/siglip-so400m-patch14-384/resolve/main/model.safetensors?filename=image-encoding-siglip-so-400m.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[SiglipVisionModel]:
        """
        :return: The model class.
        """
        from transformers.models.siglip.modeling_siglip import SiglipVisionModel
        return SiglipVisionModel # type: ignore[no-any-return]

    @classmethod
    def get_config_class(cls) -> Type[SiglipVisionConfig]:
        """
        :return: The configuration class.
        """
        from transformers.models.siglip.modeling_siglip import SiglipVisionConfig
        return SiglipVisionConfig # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        :return: The default configuration.
        """
        return {
            "hidden_size": 1152,
            "image_size": 384,
            "intermediate_size": 4304,
            "model_type": "siglip_vision_model",
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "patch_size": 14
        }
