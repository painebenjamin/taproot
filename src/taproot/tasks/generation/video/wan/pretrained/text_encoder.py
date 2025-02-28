from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from ..model import T5Encoder

__all__ = ["PretrainedWanTextEncoder"]

class PretrainedWanTextEncoder(PretrainedModelMixin):
    """
    Wan text encoder module (multilingual T5).
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-umt5-xxl.bf16.safetensors"
    dtype = "bfloat16"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return dict(
            vocab=256384,
            dim=4096,
            dim_attn=4096,
            dim_ffn=10240,
            num_heads=64,
            num_layers=24,
            num_buckets=32,
            shared_pos=False,
            dropout=0.1
        )

    @classmethod
    def get_model_class(cls) -> Type[T5Encoder]:
        """
        Returns the model class.
        """
        from ..model import T5Encoder
        return T5Encoder
