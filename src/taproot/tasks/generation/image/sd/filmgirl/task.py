from ..base import StableDiffusionBase
from .unet import StableDiffusionFilmGirlUltraUNet
from .text_encoder import StableDiffusionFilmGirlUltraTextEncoder

__all__ = ["StableDiffusionFilmGirlUltra"]

class StableDiffusionFilmGirlUltra(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-filmgirl-ultra"
    display_name = "FilmGirl Ultra Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionFilmGirlUltraUNet,
            "text_encoder": StableDiffusionFilmGirlUltraTextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "LEOSAM"
    finetune_author_url = "https://civitai.com/user/LEOSAM"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/367245"
    license_attribution = False
    license_copy_left = False
    license_derivatives = False
    license_commercial = False
    license_hosting = False
