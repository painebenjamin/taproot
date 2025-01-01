from ..base import StableDiffusionBase
from .unet import (
    StableDiffusionRealisticVisionV51UNet,
    StableDiffusionRealisticVisionV60UNet
)
from .text_encoder import (
    StableDiffusionRealisticVisionV51TextEncoder,
    StableDiffusionRealisticVisionV60TextEncoder
)

__all__ = ["StableDiffusionRealisticVisionV60"]

class StableDiffusionRealisticVisionV51(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-realistic-vision-v5-1"
    display_name = "Realistic Vision V5.1 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionRealisticVisionV51UNet,
            "text_encoder": StableDiffusionRealisticVisionV51TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "SG_161222"
    finetune_author_url = "https://civitai.com/user/SG_161222"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/130072"
    license_attribution = True
    license_copy_left = True
    license_derivatives = False
    license_commercial = True
    license_hosting = True

class StableDiffusionRealisticVisionV60(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-realistic-vision-v6-0"
    display_name = "Realistic Vision V6.0 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionRealisticVisionV60UNet,
            "text_encoder": StableDiffusionRealisticVisionV60TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "SG_161222"
    finetune_author_url = "https://civitai.com/user/SG_161222"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/245592"
    license_attribution = True
    license_copy_left = True
    license_derivatives = False
    license_commercial = True
    license_hosting = True
