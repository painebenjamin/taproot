from __future__ import annotations

from ..base import StableDiffusionXLBase
from .unet import SDXLHelloWorldV7UNet
from .text_encoder import (
    SDXLHelloWorldV7TextEncoderPrimary,
    SDXLHelloWorldV7TextEncoderSecondary,
)

__all__ = ["StableDiffusionXLHelloWorldV7"]

class StableDiffusionXLHelloWorldV7(StableDiffusionXLBase):
    task = "image-generation"
    model = "stable-diffusion-xl-helloworld-v7"
    default = False
    display_name = "LEOSAM's HelloWorld XL Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLHelloWorldV7UNet,
            "text_encoder": SDXLHelloWorldV7TextEncoderPrimary,
            "text_encoder_2": SDXLHelloWorldV7TextEncoderSecondary,
        }
    }

    """Authorship Metadata"""
    finetine_author = "LEOSAM"
    finetune_author_url = "https://civitai.com/user/LEOSAM"

    """Licensing Metadata"""
    license = "OpenRAIL++-M License with Restrictions"
    license_url = "https://civitai.com/models/license/570138"
    license_attribution = True
    license_commercial = False
    license_derivatives = False
    license_copy_left = False
    license_hosting = False
