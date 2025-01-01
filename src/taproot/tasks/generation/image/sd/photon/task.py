from ..base import StableDiffusionBase
from .unet import StableDiffusionPhotonV1UNet
from .text_encoder import StableDiffusionPhotonV1TextEncoder

__all__ = ["StableDiffusionPhotonV1"]

class StableDiffusionPhotonV1(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-photon-v1"
    display_name = "Photon V1 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionPhotonV1UNet,
            "text_encoder": StableDiffusionPhotonV1TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "Photographer"
    finetune_author_url = "https://civitai.com/user/Photographer"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/900072"
    license_attribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = True
