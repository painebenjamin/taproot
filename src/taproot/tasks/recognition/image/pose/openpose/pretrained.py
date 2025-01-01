from __future__ import annotations

from typing import Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from .helper import OpenposeDetector # type: ignore[attr-defined]

__all__ = ["PretrainedOpenposeDetector"]

class PretrainedOpenposeDetector(PretrainedModelMixin):
    """
    Pretrained OpenPose Detector
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/pose-detection-openpose.fp16.safetensors"
    load_path = "modules"

    @classmethod
    def get_model_class(cls) -> Type[OpenposeDetector]:
        """
        Get the model class
        """
        from .helper import OpenposeDetector # type: ignore[attr-defined]
        return OpenposeDetector # type: ignore[no-any-return]
