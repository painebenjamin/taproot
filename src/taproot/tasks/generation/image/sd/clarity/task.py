from ..base import StableDiffusionBase
from .unet import StableDiffusionClarityV3UNet
from .text_encoder import StableDiffusionClarityV3TextEncoder

__all__ = ["StableDiffusionClarityV3"]

class StableDiffusionClarityV3(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-clarity-v3"
    display_name = "Clarity V3 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionClarityV3UNet,
            "text_encoder": StableDiffusionClarityV3TextEncoder
        },
    }

    """Author Metadata"""
    finetune_author = "ndimensional"
    finetune_author_url = "https://civitai.com/user/ndimensional"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/142125"
    license_attribution = True
    license_derivatives = False
    licene_hosting = False
    license_commercial = True
    license_copy_left = False
