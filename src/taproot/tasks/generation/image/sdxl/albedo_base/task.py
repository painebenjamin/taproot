from __future__ import annotations

from ..base import StableDiffusionXLBase
from .unet import SDXLAlbedoBaseV31UNet
from .text_encoder import (
    SDXLAlbedoBaseV31TextEncoderPrimary,
    SDXLAlbedoBaseV31TextEncoderSecondary,
)

__all__ = ["StableDiffusionXLAlbedoBaseV31"]

class StableDiffusionXLAlbedoBaseV31(StableDiffusionXLBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-albedobase-v3-1"
    default = False
    display_name = "AlbedoBase XL V3.1 Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLAlbedoBaseV31UNet,
            "text_encoder": SDXLAlbedoBaseV31TextEncoderPrimary,
            "text_encoder_2": SDXLAlbedoBaseV31TextEncoderSecondary
        }
    }

    """Authorship Metadata"""
    finetine_author = "albedobond"
    finetune_author_url = "https://civitai.com/user/albedobond"

    """Licensing Metadata"""
    license = "OpenRAIL++-M License with Restrictions"
    license_url = "https://civitai.com/models/license/1041855"
    license_attribution = True
    license_commercial = True
    license_derivatives = False
    license_copy_left = True
    license_hosting = True
