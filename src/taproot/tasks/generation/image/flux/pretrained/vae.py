from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

__all__ = ["FluxVAE"]

class FluxVAE(PretrainedModelMixin):
    """
    The primary model for the FLUX VAE.
    """
    dtype = "bfloat16"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-vae.bf16.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[AutoencoderKL]:
        """
        Get the model class for the FLUX VAE.
        """
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
        return AutoencoderKL # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the FLUX VAE.
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
            "mid_block_add_attention": True,
            "norm_num_groups": 32,
            "out_channels": 3,
            "sample_size": 1024,
            "scaling_factor": 0.3611,
            "shift_factor": 0.1159,
            "up_block_types": ["UpDecoderBlock2D"] * 4,
            "use_post_quant_conv": False,
            "use_quant_conv": False
        }
