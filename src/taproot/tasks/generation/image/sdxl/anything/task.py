from __future__ import annotations

from ..base import StableDiffusionXLBase
from .unet import SDXLAnythingUNet
from .text_encoder import (
    SDXLAnythingTextEncoderPrimary,
    SDXLAnythingTextEncoderSecondary,
)

__all__ = ["StableDiffusionXLAnything"]

class StableDiffusionXLAnything(StableDiffusionXLBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-anything"
    default = False
    display_name = "Anything XL Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLAnythingUNet,
            "text_encoder": SDXLAnythingTextEncoderPrimary,
            "text_encoder_2": SDXLAnythingTextEncoderSecondary,
        }
    }

    """Authorship Metadata"""
    finetine_author = "Yuno779"
    finetune_author_url = "https://civitai.com/user/Yuno779"
