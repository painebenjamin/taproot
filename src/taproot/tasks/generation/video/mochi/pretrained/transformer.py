from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.transformers.transformer_mochi import MochiTransformer3DModel

__all__ = [
    "MochiTransformer",
    "MochiTransformerInt8",
    "MochiTransformerNF4"
]

class MochiTransformer(PretrainedModelMixin):
    """
    Mochi Transformer model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-mochi-v1-preview-transformer.bf16.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "activation_fn": "swiglu",
            "attention_head_dim": 128,
            "in_channels": 12,
            "max_sequence_length": 256,
            "num_attention_heads": 24,
            "num_layers": 48,
            "out_channels": None,
            "patch_size": 2,
            "pooled_projection_dim": 1536,
            "qk_norm": "rms_norm",
            "text_embed_dim": 4096,
            "time_embed_dim": 256
        }

    @classmethod
    def get_model_class(cls) -> Type[MochiTransformer3DModel]:
        """
        Returns the model class.
        """
        from diffusers.models.transformers.transformer_mochi import MochiTransformer3DModel
        return MochiTransformer3DModel # type: ignore[no-any-return]

class MochiTransformerInt8(MochiTransformer):
    """
    LTX Video Transformer model with int8 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-mochi-v1-preview-transformer.int8.bf16.safetensors"
    quantization = "bitsandbytes_8bit"
    pre_quantized = True

class MochiTransformerNF4(MochiTransformer):
    """
    LTX Video Transformer model with NF4 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-mochi-v1-preview-transformer.nf4.bf16.safetensors"
    quantization = "bitsandbytes_nf4"
    pre_quantized = True
