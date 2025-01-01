from ..base import StableDiffusionBase
from .unet import StableDiffusionEpicPhotogasmUltimateFidelityUNet
from .text_encoder import StableDiffusionEpicPhotogasmUltimateFidelityTextEncoder

__all__ = ["StableDiffusionEpicPhotogasmUltimateFidelity"]

class StableDiffusionEpicPhotogasmUltimateFidelity(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-epicphotogasm-ultimate-fidelity"
    display_name = "epiCPhotoGasm Ultimate Fidelity Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionEpicPhotogasmUltimateFidelityUNet,
            "text_encoder": StableDiffusionEpicPhotogasmUltimateFidelityTextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "epinikion"
    finetune_author_url = "https://civitai.com/user/epinikion"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/429454"
    license_attribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = True
