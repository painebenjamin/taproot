from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    import torch
    from torchvision.models import Inception3 # type: ignore[import-not-found,import-untyped,unused-ignore]

__all__ = ["PretrainedInception3"]

class PretrainedInception3(PretrainedModelMixin):
    """
    Pretrained Inception3 model for image similarity tasks.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-similarity-inception.fp16.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration.
        """
        return {
            "init_weights": False
        }

    @classmethod
    def get_model_class(cls) -> Type[Inception3]:
        """
        Get the model class.
        """
        from torchvision.models import Inception3
        return Inception3 # type: ignore[no-any-return]

    @classmethod
    def post_load_hook(cls, model: Inception3) -> None:
        """
        Post-load hook replaces the fully-connected layer with identity.
        """
        import torch
        model.fc = torch.nn.Identity()
