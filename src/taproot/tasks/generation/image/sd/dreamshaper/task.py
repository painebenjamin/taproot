from ..base import StableDiffusionBase
from .unet import StableDiffusionDreamShaperV8UNet
from .text_encoder import StableDiffusionDreamShaperV8TextEncoder

__all__ = ["StableDiffusionDreamShaperV8"]

class StableDiffusionDreamShaperV8(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-dreamshaper-v8"
    display_name = "DreamShaper V8 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionDreamShaperV8UNet,
            "text_encoder": StableDiffusionDreamShaperV8TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "Lykon"
    finetune_author_url = "https://civitai.com/user/Lykon"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/128713"
    license_attribution = False
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = True
