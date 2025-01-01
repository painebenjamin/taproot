from ..base import StableDiffusionBase
from .unet import StableDiffusionDarkSushiMixV225DUNet
from .text_encoder import StableDiffusionDarkSushiMixV225DTextEncoder

__all__ = ["StableDiffusionDarkSushiMixV225D"]

class StableDiffusionDarkSushiMixV225D(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-dark-sushi-mix-v2-25d"
    display_name = "Dark Sushi Mix V2 2.5D Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionDarkSushiMixV225DUNet,
            "text_encoder": StableDiffusionDarkSushiMixV225DTextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "Aitasai"
    finetune_author_url = "https://civitai.com/user/Aitasai"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/93208"
    license_attribution = False
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = True
