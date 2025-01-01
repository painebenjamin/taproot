from ..base import StableDiffusionBase
from .unet import StableDiffusionReVAnimatedV2UNet
from .text_encoder import StableDiffusionReVAnimatedV2TextEncoder

__all__ = ["StableDiffusionReVAnimatedV2"]

class StableDiffusionReVAnimatedV2(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-rev-animated-v2"
    display_name = "ReV Animated V2 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionReVAnimatedV2UNet,
            "text_encoder": StableDiffusionReVAnimatedV2TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "Zovya"
    finetune_author_url = "https://civitai.com/user/Zovya"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/425083"
    license_attribution = True
    license_copy_left = True
    license_derivatives = False
    license_commercial = True
    license_hosting = False
