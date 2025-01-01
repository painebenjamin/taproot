from ..base import StableDiffusionBase
from .unet import StableDiffusionMeinaMixV12UNet
from .text_encoder import StableDiffusionMeinaMixV12TextEncoder

__all__ = ["StableDiffusionMeinaMixV12"]

class StableDiffusionMeinaMixV12(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-meinamix-v12"
    display_name = "MeinaMix V12 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionMeinaMixV12UNet,
            "text_encoder": StableDiffusionMeinaMixV12TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "Meina"
    finetune_author_url = "https://civitai.com/user/Meina"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/948574"
    license_attribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = False
