from __future__ import annotations

from ..base import StableDiffusionXLBase
from .unet import SDXLAnimagineV31UNet
from .text_encoder import (
    SDXLAnimagineV31TextEncoderPrimary,
    SDXLAnimagineV31TextEncoderSecondary,
)

__all__ = ["StableDiffusionXLAnimagineV31"]

class StableDiffusionXLAnimagineV31(StableDiffusionXLBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-animagine-v3-1"
    default = False
    display_name = "Animagine XL V3.1 Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLAnimagineV31UNet,
            "text_encoder": SDXLAnimagineV31TextEncoderPrimary,
            "text_encoder_2": SDXLAnimagineV31TextEncoderSecondary,
        }
    }

    """Authorship Metadata"""
    finetine_author = "CagliostroLab"
    finetune_author_url = "https://civitai.com/user/CagliostroLab"

    """Licensing Metadata"""
    license = "OpenRAIL++-M License with Restrictions"
    license_url = "https://civitai.com/models/license/403131"
    license_attribution = True
    license_commercial = True
    license_derivatives = False
    license_copy_left = True
    license_hosting = True
