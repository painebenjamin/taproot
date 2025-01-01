from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX

__all__ = [
    "CogVideoXVAE",
    "CogVideoXVAEBF16",
    "CogVideoXVAE5B",
    "CogVideoXVAE5BBF16",
    "CogVideoX15VAE5B",
    "CogVideoX15VAE5BBF16"
]

class CogVideoXVAE(PretrainedModelMixin):
    """
    The model for the Cog Video VAE.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-vae.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[AutoencoderKLCogVideoX]:
        """
        Get the model class for the Cog Video VAE.
        """
        from diffusers.models.autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
        return AutoencoderKLCogVideoX # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Cog Video VAE.
        """
        return {
            "act_fn": "silu",
            "block_out_channels": [
                128,
                256,
                256,
                512
            ],
            "down_block_types": [
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D"
            ],
            "force_upcast": True,
            "in_channels": 3,
            "latent_channels": 16,
            "latents_mean": None,
            "latents_std": None,
            "layers_per_block": 3,
            "norm_eps": 1e-06,
            "norm_num_groups": 32,
            "out_channels": 3,
            "sample_height": 480,
            "sample_width": 720,
            "scaling_factor": 1.15258426,
            "shift_factor": None,
            "temporal_compression_ratio": 4,
            "up_block_types": [
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D"
            ],
            "use_post_quant_conv": False,
            "use_quant_conv": False,
            "invert_scale_latents": False
        }

class CogVideoXVAEBF16(CogVideoXVAE):
    """
    The model for the Cog Video VAE with BF16 precision.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-vae.bf16.safetensors"

class CogVideoXVAE5B(CogVideoXVAE):
    """
    The model for the Cog VideoX VAE for the 5B models. These reduce the scaling factor to 0.7.
    """
    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Cog Video 5 VAE.
        """
        config = super().get_default_config()
        assert isinstance(config, dict), "config must be a dictionary"
        config["scaling_factor"] = 0.7
        return config

class CogVideoXVAE5BBF16(CogVideoXVAE5B):
    """
    The model for the Cog Video 5 VAE with BF16 precision.
    """
    model_url = CogVideoXVAEBF16.model_url

class CogVideoX15VAE5B(CogVideoXVAE5B):
    """
    The model for the Cog Video 15 VAE.
    """
    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Cog Video 15 VAE.
        """
        config = super().get_default_config()
        assert isinstance(config, dict), "config must be a dictionary"
        config["invert_scale_latents"] = True
        return config

class CogVideoX15VAE5BBF16(CogVideoX15VAE5B):
    """
    The model for the Cog Video 15 VAE with BF16 precision.
    """
    model_url = CogVideoXVAE5BBF16.model_url
