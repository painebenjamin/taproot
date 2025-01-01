from ..base import StableDiffusionBase
from .unet import StableDiffusionGhostMixV2UNet
from .text_encoder import StableDiffusionGhostMixV2TextEncoder

__all__ = ["StableDiffusionGhostMixV2"]

class StableDiffusionGhostMixV2(StableDiffusionBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5-ghostmix-v2"
    display_name = "GhostMix V2 Image Generation"
    pretrained_models = {
        **StableDiffusionBase.pretrained_models,
        **{
            "unet": StableDiffusionGhostMixV2UNet,
            "text_encoder": StableDiffusionGhostMixV2TextEncoder
        },
    }

    """Authorship Metadata"""
    finetune_author = "_GhostInShell_"
    finetune_author_url = "https://civitai.com/user/_GhostInShell_"

    """License Metadata"""
    license = "OpenRAIL-M License with Restrictions"
    license_url = "https://civitai.com/models/license/76907"
    license_attribution = True
    license_copy_left = True
    license_derivatives = False
    license_commercial = False
    license_hosting = True
