from ..base import StableDiffusionBase
from .unet import StableDiffusionChilloutMixNiUNet
from .text_encoder import StableDiffusionChilloutMixNiTextEncoder

__all__ = ["StableDiffusionChilloutMixNi"]

class StableDiffusionChilloutMixNi(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-chillout-mix-ni"
    display_name = "Chillout Mix Ni Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionChilloutMixNiUNet,
            "text_encoder": StableDiffusionChilloutMixNiTextEncoder
        },
    }

    """Author Metadata"""
    finetune_author = "Dreamlike Art"
    finetune_author_url = "https://dreamlike.art"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0/blob/main/LICENSE.md"
    license_attribution = True
    license_copy_left = True
    license_derivatives = False
    license_hosting = False
    license_commercial = False
