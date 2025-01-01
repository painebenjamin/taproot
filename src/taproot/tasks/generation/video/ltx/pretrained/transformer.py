from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel

__all__ = [
    "LTXVideoTransformer",
    "LTXVideoTransformerInt8",
    "LTXVideoTransformerNF4"
]

class LTXVideoTransformer(PretrainedModelMixin):
    """
    LTX Video Transformer model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-ltx-transformer.bf16.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "activation_fn": "gelu-approximate",
            "attention_bias": True,
            "attention_head_dim": 64,
            "attention_out_bias": True,
            "caption_channels": 4096,
            "cross_attention_dim": 2048,
            "in_channels": 128,
            "norm_elementwise_affine": False,
            "norm_eps": 1e-06,
            "num_attention_heads": 32,
            "num_layers": 28,
            "out_channels": 128,
            "patch_size": 1,
            "patch_size_t": 1,
            "qk_norm": "rms_norm_across_heads"
        }

    @classmethod
    def get_model_class(cls) -> Type[LTXVideoTransformer3DModel]:
        """
        Returns the model class.
        """
        from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel
        return LTXVideoTransformer3DModel # type: ignore[no-any-return]

class LTXVideoTransformerInt8(LTXVideoTransformer):
    """
    LTX Video Transformer model with int8 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-ltx-transformer.int8.bf16.safetensors"
    quantization = "bitsandbytes_8bit"
    pre_quantized = True

class LTXVideoTransformerNF4(LTXVideoTransformer):
    """
    LTX Video Transformer model with NF4 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-ltx-transformer.nf4.bf16.safetensors"
    quantization = "bitsandbytes_4bit"
    pre_quantized = True
