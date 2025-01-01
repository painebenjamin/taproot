from ..base import StableDiffusionBase
from .unet import StableDiffusionMajicMixRealisticV7UNet
from .text_encoder import StableDiffusionMajicMixRealisticV7TextEncoder

__all__ = ["StableDiffusionMajicMixRealisticV7"]

class StableDiffusionMajicMixRealisticV7(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-majicmix-realistic-v7"
    display_name = "MajicMix Realistic V7 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionMajicMixRealisticV7UNet,
            "text_encoder": StableDiffusionMajicMixRealisticV7TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "Merjic"
    finetune_author_url = "https://civitai.com/user/Merjic"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/176425"
    license_attribution = False
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = True
