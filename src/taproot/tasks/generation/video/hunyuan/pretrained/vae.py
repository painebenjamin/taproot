from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo

__all__ = ["HunyuanVideoVAE"]

class HunyuanVideoVAE(PretrainedModelMixin):
    """
    The primary model for the Hunyuan Video VAE.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-hunyuan-vae.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[AutoencoderKLHunyuanVideo]:
        """
        Get the model class for the Hunyuan Video VAE.
        """
        from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
        return AutoencoderKLHunyuanVideo # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Hunyuan Video VAE.
        """
        return {
            "block_out_channels": [128, 256, 512, 512],
            "down_block_types": ["HunyuanVideoDownBlock3D"] * 4,
            "in_channels": 3,
            "latent_channels": 16,
            "layers_per_block": 2,
            "mid_block_add_attention": True,
            "norm_num_groups": 32,
            "out_channels": 3,
            "scaling_factor": 0.476986,
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 4,
            "up_block_types": ["HunyuanVideoUpBlock3D"] * 4,
        }
