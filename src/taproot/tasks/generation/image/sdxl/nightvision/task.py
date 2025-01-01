from __future__ import annotations

from ..base import StableDiffusionXLBase
from .unet import SDXLNightVisionV9UNet
from .text_encoder import (
    SDXLNightVisionV9TextEncoderPrimary,
    SDXLNightVisionV9TextEncoderSecondary,
)

__all__ = ["StableDiffusionXLNightVisionV9"]

class StableDiffusionXLNightVisionV9(StableDiffusionXLBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-nightvision-v9"
    default = False
    display_name = "NightVision XL V9 Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLNightVisionV9UNet,
            "text_encoder": SDXLNightVisionV9TextEncoderPrimary,
            "text_encoder_2": SDXLNightVisionV9TextEncoderSecondary,
        }
    }

    """Authorship Metadata"""
    finetine_author = "socalguitarist"
    finetune_author_url = "https://civitai.com/user/socalguitarist"

    """Licensing Metadata"""
    license = "OpenRAIL++-M License with Restrictions"
    license_url = "https://civitai.com/models/license/577919"
    license_attribution = False
    license_commercial = True
    license_derivatives = False
    license_copy_left = False
    license_hosting = False
