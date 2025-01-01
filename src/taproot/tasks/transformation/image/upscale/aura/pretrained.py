from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from .model import AuraSR # type: ignore[attr-defined]

__all__ = [
    "PretrainedAuraSR",
    "PretrainedAuraSRV2"
]

class PretrainedAuraSR(PretrainedModelMixin):
    """
    Pretrained model for Aura Super Resolution
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/super-resolution-aura.fp16.safetensors"
    load_path = "upscaler"
    spread_config = False

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for this pretrained model
        """
        return {
            "style_network": {
                "dim_in": 128,
                "dim_out": 512,
                "depth": 4
            },
            "dim": 64,
            "image_size": 256,
            "input_image_size": 64,
            "unconditional": True,
            "skip_connect_scale": 0.4
        }

    @classmethod
    def get_model_class(cls) -> Type[AuraSR]:
        """
        Get the model class for this pretrained model
        """
        from .model import AuraSR # type: ignore[attr-defined]
        return AuraSR # type: ignore[no-any-return]

class PretrainedAuraSRV2(PretrainedAuraSR):
    """
    Pretrained model for Aura Super Resolution V2
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/super-resolution-aura-v2.fp16.safetensors"
