from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

__all__ = ["StableDiffusionVAE"]

class StableDiffusionVAE(PretrainedModelMixin):
    """
    The primary model for the Stable Diffusion VAE.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-vae.fp16.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[AutoencoderKL]:
        """
        Get the model class for the Stable Diffusion VAE.
        """
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
        return AutoencoderKL # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Stable Diffusion VAE.
        """
        return {
            "act_fn": "silu",
            "down_block_types": ["DownEncoderBlock2D"] * 4,
            "up_block_types": ["UpDecoderBlock2D"] * 4,
            "block_out_channels": [128, 256, 512, 512],
            "in_channels": 3,
            "latent_channels": 4,
            "layers_per_block": 2,
            "norm_num_groups": 32,
            "out_channels": 3,
            "sample_size": 512,
        }
