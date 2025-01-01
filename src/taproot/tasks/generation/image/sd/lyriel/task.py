from ..base import StableDiffusionBase
from .unet import StableDiffusionLyrielV16UNet
from .text_encoder import StableDiffusionLyrielV16TextEncoder

__all__ = ["StableDiffusionLyrielV16"]

class StableDiffusionLyrielV16(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-lyriel-v1-6"
    display_name = "Lyriel V1.6 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionLyrielV16UNet,
            "text_encoder": StableDiffusionLyrielV16TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "Lyriel"
    finetune_author_url = "https://civitai.com/user/Lyriel"

    """License Metadata"""
    license = "OpenRAIL-M License"
    license_url = "https://civitai.com/models/license/72396"
    license_attribution = True
    license_copy_left = False
    license_derivatives = True
    license_commercial = True
    license_hosting = True
