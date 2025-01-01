from __future__ import annotations

from typing import Type, Dict, Any, Optional, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from .inference import MiDaSInference # type: ignore[attr-defined]

__all__ = ["PretrainedMidasDetector"]

class PretrainedMidasDetector(PretrainedModelMixin):
    """
    Pretrained MiDaS Depth Detection model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/depth-detection-midas.fp16.safetensors"
    load_path = "model"

    @classmethod
    def get_model_class(cls) -> Type[MiDaSInference]:
        """
        Get the model class
        """
        from .inference import MiDaSInference # type: ignore[attr-defined]
        return MiDaSInference # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration
        """
        return {
            "model_type": "dpt_hybrid",
            "model_path": None
        }
