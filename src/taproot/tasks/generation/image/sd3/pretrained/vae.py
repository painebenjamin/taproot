from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

__all__ = ["StableDiffusion3VAE"]

class StableDiffusion3VAE(PretrainedModelMixin):
    """
    The primary model for the Stable Diffusion 3 VAE.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v3-vae.fp16.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[AutoencoderKL]:
        """
        Get the model class for the Stable Diffusion 3 VAE.
        """
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
        return AutoencoderKL # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Stable Diffusion 3 VAE.
        """
        return {
            "act_fn": "silu",
            "block_out_channels": [128, 256, 512, 512],
            "down_block_types": ["DownEncoderBlock2D"] * 4, 
            "force_upcast": True,
            "in_channels": 3,
            "latent_channels": 16,
            "latents_mean": None,
            "latents_std": None,
            "layers_per_block": 2,
            "norm_num_groups": 32,
            "out_channels": 3,
            "sample_size": 1024,
            "scaling_factor": 1.5305,
            "shift_factor": 0.0609,
            "up_block_types": ["UpDecoderBlock2D"] * 4,
            "use_post_quant_conv": False,
            "use_quant_conv": False
        }
