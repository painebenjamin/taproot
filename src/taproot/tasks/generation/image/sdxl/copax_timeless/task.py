from __future__ import annotations

from ..base import StableDiffusionXLBase
from .unet import SDXLCopaxTimeLessV13UNet
from .text_encoder import (
    SDXLCopaxTimeLessV13TextEncoderPrimary,
    SDXLCopaxTimeLessV13TextEncoderSecondary,
)

__all__ = ["StableDiffusionXLCopaxTimeLessV13"]

class StableDiffusionXLCopaxTimeLessV13(StableDiffusionXLBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-copax-timeless-v13"
    default = False
    display_name = "Copax TimeLess V13 Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLCopaxTimeLessV13UNet,
            "text_encoder": SDXLCopaxTimeLessV13TextEncoderPrimary,
            "text_encoder_2": SDXLCopaxTimeLessV13TextEncoderSecondary,
        }
    }

    """Authorship Metadata"""
    finetine_author = "copax"
    finetune_author_url = "https://civitai.com/user/copax"

    """Licensing Metadata"""
    license = "OpenRAIL++-M License with Restrictions"
    license_url = "https://civitai.com/models/license/724334"
    license_attribution = True
    license_commercial = True
    license_derivatives = False
    license_copy_left = True
    license_hosting = False
