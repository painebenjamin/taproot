from __future__ import annotations

from ..base import StableDiffusionXLBase
from .unet import SDXLUnstableDiffusersNihilmaniaUNet
from .text_encoder import (
    SDXLUnstableDiffusersNihilmaniaTextEncoderPrimary,
    SDXLUnstableDiffusersNihilmaniaTextEncoderSecondary,
)

__all__ = ["StableDiffusionXLUnstableDiffusersNihilmania"]

class StableDiffusionXLUnstableDiffusersNihilmania(StableDiffusionXLBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-unstable-diffusers-nihilmania"
    default = False
    display_name = "SDXL Unstable Diffusers NihilMania Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLUnstableDiffusersNihilmaniaUNet,
            "text_encoder": SDXLUnstableDiffusersNihilmaniaTextEncoderPrimary,
            "text_encoder_2": SDXLUnstableDiffusersNihilmaniaTextEncoderSecondary,
        }
    }

    """Authorship Metadata"""
    finetine_author = "Yamer"
    finetune_author_url = "https://civitai.com/user/Yamer"

    """Licensing Metadata"""
    license = "OpenRAIL++-M License with Restrictions"
    license_url = "https://civitai.com/models/license/395107"
    license_attribution = True
    license_commercial = True
    license_derivatives = False
    license_copy_left = False
    license_hosting = False
