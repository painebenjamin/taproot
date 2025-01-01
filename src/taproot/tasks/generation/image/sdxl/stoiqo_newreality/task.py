from __future__ import annotations

from ..base import StableDiffusionXLBase
from .unet import SDXLStoiqoNewRealityProUNet
from .text_encoder import (
    SDXLStoiqoNewRealityProTextEncoderPrimary,
    SDXLStoiqoNewRealityProTextEncoderSecondary,
)

__all__ = ["StableDiffusionXLStoiqoNewRealityPro"]

class StableDiffusionXLStoiqoNewRealityPro(StableDiffusionXLBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-stoiqo-newreality-pro"
    default = False
    display_name = "Stoiqo New Reality XL Pro Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLStoiqoNewRealityProUNet,
            "text_encoder": SDXLStoiqoNewRealityProTextEncoderPrimary,
            "text_encoder_2": SDXLStoiqoNewRealityProTextEncoderSecondary,
        }
    }

    """Authorship Metadata"""
    finetine_author = "ALIENHAZE"
    finetune_author_url = "https://civitai.com/user/ALIENHAZE"

    """Licensing Metadata"""
    license = "OpenRAIL++-M License with Restrictions"
    license_url = "https://civitai.com/models/license/690310"
    license_attribution = False
    license_commercial = True
    license_derivatives = False
    license_copy_left = False
    license_hosting = False
