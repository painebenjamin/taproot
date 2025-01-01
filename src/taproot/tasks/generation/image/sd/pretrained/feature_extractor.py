from __future__ import annotations

from taproot.util import PretrainedModelMixin

from typing import Optional, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import CLIPImageProcessor # type: ignore[import-untyped]

__all__ = [
    "StableDiffusionFeatureExtractor"
]

class StableDiffusionFeatureExtractor(PretrainedModelMixin):
    """
    The model for the Stable Diffusion FeatureExtractor.
    """

    @classmethod
    def get_model_class(cls) -> Type[CLIPImageProcessor]:
        """
        Get the model class for the Stable Diffusion FeatureExtractor.
        """
        from transformers import CLIPImageProcessor
        return CLIPImageProcessor # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Stable Diffusion FeatureExtractor.
        """
        return {
            "crop_size": 224,
            "do_center_crop": True,
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_resize": True,
            "feature_extractor_type": "CLIPImageProcessor",
            "image_mean": [
                0.48145466,
                0.4578275,
                0.40821073
            ],
            "image_std": [
                0.26862954,
                0.26130258,
                0.27577711
            ],
            "resample": 3,
            "size": 224
        }
