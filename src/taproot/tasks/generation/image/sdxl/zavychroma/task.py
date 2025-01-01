from __future__ import annotations

from ..base import StableDiffusionXLBase
from .unet import SDXLZavyChromaV10UNet
from .text_encoder import (
    SDXLZavyChromaV10TextEncoderPrimary,
    SDXLZavyChromaV10TextEncoderSecondary,
)

__all__ = ["StableDiffusionXLZavyChromaV10"]

class StableDiffusionXLZavyChromaV10(StableDiffusionXLBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-zavychroma-v10"
    default = False
    display_name = "ZavyChromaXL V10 Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLZavyChromaV10UNet,
            "text_encoder": SDXLZavyChromaV10TextEncoderPrimary,
            "text_encoder_2": SDXLZavyChromaV10TextEncoderSecondary,
        }
    }

    """Authorship Metadata"""
    finetine_author = "Zavy"
    finetune_author_url = "https://civitai.com/user/Zavy"

    """Licensing Metadata"""
    license = "OpenRAIL++-M License with Restrictions"
    license_url = "https://civitai.com/models/license/916744"
    license_attribution = False
    license_commercial = True
    license_derivatives = False
    license_copy_left = True
    license_hosting = False
