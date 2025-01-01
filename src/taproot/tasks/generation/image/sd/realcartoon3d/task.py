from ..base import StableDiffusionBase
from .unet import StableDiffusionRealCartoon3DV17UNet
from .text_encoder import StableDiffusionRealCartoon3DV17TextEncoder

__all__ = ["StableDiffusionRealCartoon3DV17"]

class StableDiffusionRealCartoon3DV17(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-realcartoon3d-v17"
    display_name = "RealCartoon3D V17 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionRealCartoon3DV17UNet,
            "text_encoder": StableDiffusionRealCartoon3DV17TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "7whitefire7"
    finetune_author_url = "https://civitai.com/user/7whitefire7"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/637156"
    license_attribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = True
