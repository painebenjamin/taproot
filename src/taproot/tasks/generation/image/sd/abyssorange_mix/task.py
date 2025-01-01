from ..base import StableDiffusionBase
from .unet import StableDiffusionAbyssOrangeMixV3UNet
from .text_encoder import StableDiffusionAbyssOrangeMixV3TextEncoder

__all__ = ["StableDiffusionAbyssOrangeMixV3"]

class StableDiffusionAbyssOrangeMixV3(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-abyssorange-mix-v3"
    display_name = "AbyssOrange Mix V3 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionAbyssOrangeMixV3UNet,
            "text_encoder": StableDiffusionAbyssOrangeMixV3TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "liudinglin"
    finetune_author_url = "https://civitai.com/user/liudinglin"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/17233"
    license_attribution = True
    license_copy_left = True
    license_derivatives = False
    license_commercial = False
    license_hosting = False
