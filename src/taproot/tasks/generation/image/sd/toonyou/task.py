from ..base import StableDiffusionBase
from .unet import StableDiffusionToonYouBetaV6UNet
from .text_encoder import StableDiffusionToonYouBetaV6TextEncoder

__all__ = ["StableDiffusionToonYouBetaV6"]

class StableDiffusionToonYouBetaV6(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-toonyou-beta-v6"
    display_name = "ToonYou Beta V6 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionToonYouBetaV6UNet,
            "text_encoder": StableDiffusionToonYouBetaV6TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "Bradcatt"
    finetune_author_url = "https://civitai.com/user/Bradcatt"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/125771"
    license_attribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = False
