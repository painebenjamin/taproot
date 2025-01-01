from __future__ import annotations

from typing import Optional, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from .model import IFNet

__all__ = ["RIFENetModel"]

class RIFENetModel(PretrainedModelMixin):
    """
    Pretrained model for RIFE (Real-Time Intermediate Flow Estimation) model.
    """
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-interpolation-rife-flownet.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[IFNet]:
        """
        :return: The model class that this PretrainedModelMixin is wrapping.
        """
        from .model import IFNet
        return IFNet
