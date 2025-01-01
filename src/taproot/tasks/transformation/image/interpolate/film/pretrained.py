from __future__ import annotations

from typing import Optional, Union, List
from taproot.util import PretrainedModelMixin

__all__ = ["FILMNetModel"]

class FILMNetModel(PretrainedModelMixin):
    """
    Pretrained model for FILMNet
    """
    model_url: Optional[Union[str, List[str]]] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-interpolation-film-net.fp16.pt"
    use_torch_jit: bool = True
