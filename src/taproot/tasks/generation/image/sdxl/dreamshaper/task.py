from __future__ import annotations

from ..base import StableDiffusionXLBase
from .unet import SDXLDreamShaperAlphaV2UNet
from .text_encoder import (
    SDXLDreamShaperAlphaV2TextEncoderPrimary,
    SDXLDreamShaperAlphaV2TextEncoderSecondary,
)

__all__ = ["StableDiffusionXLDreamShaperAlphaV2"]

class StableDiffusionXLDreamShaperAlphaV2(StableDiffusionXLBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-dreamshaper-alpha-v2"
    default = False
    display_name = "DreamShaper XL Alpha V2 Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLDreamShaperAlphaV2UNet,
            "text_encoder": SDXLDreamShaperAlphaV2TextEncoderPrimary,
            "text_encoder_2": SDXLDreamShaperAlphaV2TextEncoderSecondary,
        }
    }

    """Authorship Metadata"""
    finetine_author = "Lykon"
    finetune_author_url = "https://civitai.com/user/Lykon"

    """Licensing Metadata"""
    license = "OpenRAIL++-M License with Restrictions"
    license_url = "https://civitai.com/models/license/126688"
    license_attribution = False
    license_commercial = True
    license_derivatives = False
    license_copy_left = False
    license_hosting = True
