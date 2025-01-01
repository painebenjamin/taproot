from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformer3DModel

__all__ = [
    "HunyuanVideoTransformer",
    "HunyuanVideoTransformerInt8",
    "HunyuanVideoTransformerNF4"
]

class HunyuanVideoTransformer(PretrainedModelMixin):
    """
    Hunyuan Video Transformer model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-hunyuan-transformer.bf16.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "attention_head_dim": 128,
            "guidance_embeds": True,
            "in_channels": 16,
            "mlp_ratio": 4.0,
            "num_attention_heads": 24,
            "num_layers": 20,
            "num_refiner_layers": 2,
            "num_single_layers": 40,
            "out_channels": 16,
            "patch_size": 2,
            "patch_size_t": 1,
            "pooled_projection_dim": 768,
            "qk_norm": "rms_norm",
            "rope_axes_dim": [16, 56, 56],
            "rope_theta": 256.0,
            "text_embed_dim": 4096
        }

    @classmethod
    def get_model_class(cls) -> Type[HunyuanVideoTransformer3DModel]:
        """
        Returns the model class.
        """
        from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformer3DModel
        return HunyuanVideoTransformer3DModel # type: ignore[no-any-return]

class HunyuanVideoTransformerInt8(HunyuanVideoTransformer):
    """
    Hunyuan Video Transformer model in int8 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-hunyuan-transformer.int8.bf16.safetensors"
    quantization = "bitsandbytes_8bit"
    pre_quantized = True

class HunyuanVideoTransformerNF4(HunyuanVideoTransformer):
    """
    Hunyuan Video Transformer model in nf4 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-hunyuan-transformer.nf4.bf16.safetensors"
    quantization = "bitsandbytes_4bit"
    pre_quantized = True
