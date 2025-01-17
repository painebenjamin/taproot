from ..base import StableDiffusionBase
from .unet import StableDiffusionSerenityV21UNet
from .text_encoder import StableDiffusionSerenityV21TextEncoder

__all__ = ["StableDiffusionSerenityV21"]

class StableDiffusionSerenityV21(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-serenity-v2-1"
    display_name = "Serenity V2.1 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionSerenityV21UNet,
            "text_encoder": StableDiffusionSerenityV21TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "malcolmrey"
    finetune_author_url = "https://civitai.com/user/malcolmrey"

    """License Metadata"""
    license = "OpenRAIL-M License with Addendum"
    license_url = "https://civitai.com/models/license/360311"
    license_attribution = False
    license_copy_left = False
    license_derivatives = True
    license_commercial = True
    license_hosting = True
