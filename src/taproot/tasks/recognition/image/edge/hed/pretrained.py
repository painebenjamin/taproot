from __future__ import annotations

from typing import Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from .model import HEDDetector # type: ignore[attr-defined]

__all__ = ["PretrainedHEDDetector"]

class PretrainedHEDDetector(PretrainedModelMixin):
    """
    Pretrained MiDaS Depth Detection model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/edge-detection-hed.fp16.safetensors"
    load_path = "netNetwork"

    @classmethod
    def get_model_class(cls) -> Type[HEDDetector]:
        """
        Get the model class
        """
        from .model import HEDDetector # type: ignore[attr-defined]
        return HEDDetector # type: ignore[no-any-return]
