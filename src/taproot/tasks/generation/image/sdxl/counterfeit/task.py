from __future__ import annotations

from ..base import StableDiffusionXLBase
from .unet import SDXLCounterfeitV25UNet
from .text_encoder import (
    SDXLCounterfeitV25TextEncoderPrimary,
    SDXLCounterfeitV25TextEncoderSecondary,
)

__all__ = ["StableDiffusionXLCounterfeitV25"]

class StableDiffusionXLCounterfeitV25(StableDiffusionXLBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-counterfeit-v2-5"
    default = False
    display_name = "CounterfeitXL V2.5 Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLCounterfeitV25UNet,
            "text_encoder": SDXLCounterfeitV25TextEncoderPrimary,
            "text_encoder_2": SDXLCounterfeitV25TextEncoderSecondary,
        }
    }

    """Authorship Metadata"""
    finetine_author = "rqdwdw"
    finetune_author_url = "https://civitai.com/user/rqdwdw"

    """Licensing Metadata"""
    license = "OpenRAIL++-M License with Restrictions"
    license_url = "https://civitai.com/models/license/265012"
    license_attribution = False
    license_commercial = True
    license_derivatives = False
    license_copy_left = True
    license_hosting = True
