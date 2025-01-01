from __future__ import annotations

from typing import Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from .net import PidiNetDetector # type: ignore[attr-defined]

__all__ = ["PretrainedPidiDetector"]

class PretrainedPidiDetector(PretrainedModelMixin):
    """
    Pretrained MiDaS Depth Detection model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/edge-detection-pidi.fp16.safetensors"
    load_path = "netNetwork"

    @classmethod
    def get_model_class(cls) -> Type[PidiNetDetector]:
        """
        Get the model class
        """
        from .net import PidiNetDetector # type: ignore[attr-defined]
        return PidiNetDetector # type: ignore[no-any-return]
