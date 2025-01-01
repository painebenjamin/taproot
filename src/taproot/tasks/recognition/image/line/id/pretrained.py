from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from functools import partial

from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from .model import Generator, UNetGenerator

__all__ = [
    "PretrainedLineartDetector",
    "PretrainedCoarseLineartDetector",
    "PretrainedAnimeLineartDetector"
]

class PretrainedLineartDetector(PretrainedModelMixin):
    """
    Pretrained Lineart Detection Model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/line-detection-informative-drawings.fp16.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the model
        """
        return {
            "input_nc": 3,
            "output_nc": 1,
            "n_residual_blocks": 3,
            "sigmoid": True
        }

    @classmethod
    def get_model_class(cls) -> Type[Generator]:
        """
        Get the model class
        """
        from .model import Generator
        return Generator

class PretrainedCoarseLineartDetector(PretrainedLineartDetector):
    """
    The coarse version of the pretrained lineart detector
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/line-detection-informative-drawings-coarse.fp16.safetensors"

class PretrainedAnimeLineartDetector(PretrainedModelMixin):
    """
    Pretrained Anime Lineart Detection Model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/line-detection-informative-drawings-anime.fp16.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the model
        """
        import torch
        return {
            "input_nc": 3,
            "output_nc": 1,
            "num_downs": 8,
            "ngf": 64,
            "use_dropout": False,
            "norm_layer": partial(
                torch.nn.InstanceNorm2d,
                affine=False,
                track_running_stats=False
            )
        }

    @classmethod
    def get_model_class(cls) -> Type[UNetGenerator]:
        """
        Get the model class
        """
        from .model import UNetGenerator
        return UNetGenerator
