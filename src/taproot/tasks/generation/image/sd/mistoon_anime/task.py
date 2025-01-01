from ..base import StableDiffusionBase
from .unet import StableDiffusionMistoonAnimeV3UNet
from .text_encoder import StableDiffusionMistoonAnimeV3TextEncoder

__all__ = ["StableDiffusionMistoonAnimeV3"]

class StableDiffusionMistoonAnimeV3(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-mistoon-anime-v3"
    display_name = "Mistoon Anime V3 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionMistoonAnimeV3UNet,
            "text_encoder": StableDiffusionMistoonAnimeV3TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "Inzaniak"
    finetune_author_url = "https://civitai.com/user/Inzaniak"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/348981"
    license_attribution = False
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = False
