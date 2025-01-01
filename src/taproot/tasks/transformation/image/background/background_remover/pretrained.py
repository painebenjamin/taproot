from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from .u2net import U2NET # type: ignore[attr-defined]

__all__ = ["PretrainedBackgroundRemover"]

class PretrainedBackgroundRemover(PretrainedModelMixin):
    """
    Pretrained BackgroundRemover model.
    """
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/background-removal-u2net.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "in_ch": 3,
            "out_ch": 1
        }

    @classmethod
    def get_model_class(cls) -> Type[U2NET]:
        """
        Returns the model class.
        """
        from .u2net import U2NET # type: ignore[attr-defined]
        return U2NET # type: ignore[no-any-return]
