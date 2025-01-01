from __future__ import annotations

from ..base import StableDiffusionXLBase
from .unet import SDXLJuggernautV11UNet
from .text_encoder import (
    SDXLJuggernautV11TextEncoderPrimary,
    SDXLJuggernautV11TextEncoderSecondary,
)

__all__ = ["StableDiffusionXLJuggernautV11"]

class StableDiffusionXLJuggernautV11(StableDiffusionXLBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-juggernaut-v11"
    default = True # This is our default for image generation for now
    display_name = "Juggernaut XL V11 Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLJuggernautV11UNet,
            "text_encoder": SDXLJuggernautV11TextEncoderPrimary,
            "text_encoder_2": SDXLJuggernautV11TextEncoderSecondary,
        }
    }

    """Authorship Metadata"""
    finetine_author = "KandooAI"
    finetune_author_url = "https://civitai.com/user/KandooAI"

    """Licensing Metadata"""
    license = "OpenRAIL++-M License with Restrictions"
    license_url = "https://civitai.com/models/license/782002"
    license_attribution = True
    license_commercial = True
    license_derivatives = False
    license_copy_left = True
    license_hosting = False
