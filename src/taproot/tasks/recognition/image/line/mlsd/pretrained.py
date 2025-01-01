from __future__ import annotations

from typing import Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from .model import MLSD

__all__ = ["PretrainedMobileLineSegmentDetector"]

class PretrainedMobileLineSegmentDetector(PretrainedModelMixin):
    """
    Pretrained MLSD model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/line-detection-mlsd.fp16.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[MLSD]:
        """
        Get the model class
        """
        from .model import MLSD
        return MLSD
