from ..base import StableDiffusionBase
from .unet import StableDiffusionEpicRealismV5UNet
from .text_encoder import StableDiffusionEpicRealismV5TextEncoder

__all__ = ["StableDiffusionEpicRealismV5"]

class StableDiffusionEpicRealismV5(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-epicrealism-v5"
    display_name = "epiCRealism V5 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionEpicRealismV5UNet,
            "text_encoder": StableDiffusionEpicRealismV5TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "epinikion"
    finetune_author_url = "https://civitai.com/user/epinikion"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/143906"
    license_attribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = True
