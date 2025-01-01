from __future__ import annotations

from ..base import StableDiffusionXLBase
from .unet import SDXLRealVisV50UNet
from .text_encoder import (
    SDXLRealVisV50TextEncoderPrimary,
    SDXLRealVisV50TextEncoderSecondary,
)

__all__ = ["StableDiffusionXLRealVisV50"]

class StableDiffusionXLRealVisV50(StableDiffusionXLBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-realvis-v5"
    default = False
    display_name = "RealVisXL V5 Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLRealVisV50UNet,
            "text_encoder": SDXLRealVisV50TextEncoderPrimary,
            "text_encoder_2": SDXLRealVisV50TextEncoderSecondary,
        }
    }

    """Authorship Metadata"""
    finetine_author = "SG_161222"
    finetune_author_url = "https://civitai.com/user/SG_161222"

    """Licensing Metadata"""
    license = "OpenRAIL++-M License with Restrictions"
    license_url = "https://civitai.com/models/license/789646"
    license_attribution = True
    license_commercial = True
    license_derivatives = False
    license_copy_left = False
    license_hosting = True
