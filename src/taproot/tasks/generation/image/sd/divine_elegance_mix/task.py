from ..base import StableDiffusionBase
from .unet import StableDiffusionDivineEleganceMixV10UNet
from .text_encoder import StableDiffusionDivineEleganceMixV10TextEncoder

__all__ = ["StableDiffusionDivineEleganceMixV10"]

class StableDiffusionDivineEleganceMixV10(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-divine-elegance-mix-v10"
    display_name = "Divine Elegance Mix V10 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionDivineEleganceMixV10UNet,
            "text_encoder": StableDiffusionDivineEleganceMixV10TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "TroubleDarkness"
    finetune_author_url = "https://civitai.com/user/TroubleDarkness"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/432048"
    license_attribution = True
    license_copy_left = True
    license_derivatives = False
    license_commercial = False
    license_hosting = False
