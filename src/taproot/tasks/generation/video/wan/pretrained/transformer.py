from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from ..model import WanModel

__all__ = [
    "PretrainedWanT2V1BTransformer",
    "PretrainedWanT2V14BTransformer",
    "PretrainedWanT2V14BTransformerQ80",
    "PretrainedWanT2V14BTransformerQ6K",
    "PretrainedWanT2V14BTransformerQ5KM",
    "PretrainedWanT2V14BTransformerQ4KM",
    "PretrainedWanT2V14BTransformerQ3KM",
]

class PretrainedWanTransformer(PretrainedModelMixin):
    """
    Pretrained Wan model base class.
    """
    @classmethod
    def get_model_class(cls) -> Type[WanModel]:
        """
        Returns the model class.
        """
        from ..model import WanModel
        return WanModel

class PretrainedWanT2V1BTransformer(PretrainedWanTransformer):
    """
    Pretrained Wan text-to-video 1.3B model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-wan-v2-1-transformer-1b.fp16.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "cross_attn_norm": True,
            "dim": 1536,
            "eps": 1e-6,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "in_dim": 16,
            "model_type": "t2v",
            "num_heads": 12,
            "num_layers": 30,
            "out_dim": 16,
            "qk_norm": True,
            "text_len": 512,
        }

class PretrainedWanT2V14BTransformer(PretrainedWanTransformer):
    """
    Pretrained Wan text-to-video 14B model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-wan-v2-1-transformer-14b.bf16.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "cross_attn_norm": True,
            "dim": 5120,
            "eps": 1e-6,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "in_dim": 16,
            "model_type": "t2v",
            "num_heads": 40,
            "num_layers": 40,
            "out_dim": 16,
            "qk_norm": True,
            "text_len": 512,
        }

class PretrainedWanT2V14BTransformerQ80(PretrainedWanT2V14BTransformer):
    """
    Pretrained Wan text-to-video 14B model with Q8-0 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-wan-v2-1-transformer-14b-q8-0.gguf"
    quantization = "gguf"
    pre_quantized = True

class PretrainedWanT2V14BTransformerQ6K(PretrainedWanT2V14BTransformer):
    """
    Pretrained Wan text-to-video 14B model with Q6-K quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-wan-v2-1-transformer-14b-q6-k.gguf"
    quantization = "gguf"
    pre_quantized = True

class PretrainedWanT2V14BTransformerQ5KM(PretrainedWanT2V14BTransformer):
    """
    Pretrained Wan text-to-video 14B model with Q5-K-M quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-wan-v2-1-transformer-14b-q5-k-m.gguf"
    quantization = "gguf"
    pre_quantized = True

class PretrainedWanT2V14BTransformerQ4KM(PretrainedWanT2V14BTransformer):
    """
    Pretrained Wan text-to-video 14B model with Q4-K-M quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-wan-v2-1-transformer-14b-q4-k-m.gguf"
    quantization = "gguf"
    pre_quantized = True

class PretrainedWanT2V14BTransformerQ3KM(PretrainedWanT2V14BTransformer):
    """
    Pretrained Wan text-to-video 14B model with Q3-K-M quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-wan-v2-1-transformer-14b-q3-k-m.gguf"
    quantization = "gguf"
    pre_quantized = True
