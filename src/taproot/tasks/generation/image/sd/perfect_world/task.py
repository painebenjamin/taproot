from ..base import StableDiffusionBase
from .unet import StableDiffusionPerfectWorldV6UNet
from .text_encoder import StableDiffusionPerfectWorldV6TextEncoder

__all__ = ["StableDiffusionPerfectWorldV6"]

class StableDiffusionPerfectWorldV6(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-perfect-world-v6"
    display_name = "Perfect World V6 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionPerfectWorldV6UNet,
            "text_encoder": StableDiffusionPerfectWorldV6TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "Bloodsuga"
    finetune_author_url = "https://civitai.com/user/Bloodsuga"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/179446"
    license_attribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = True
