from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.autoencoders.autoencoder_kl_mochi import AutoencoderKLMochi

__all__ = ["MochiVAE"]

class MochiVAE(PretrainedModelMixin):
    """
    The model for the Mochi V1 Preview VAE.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-mochi-v1-preview-vae.bf16.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[AutoencoderKLMochi]:
        """
        Get the model class for the Mochi V1 Preview VAE.
        """
        from diffusers.models.autoencoders.autoencoder_kl_mochi import AutoencoderKLMochi
        return AutoencoderKLMochi # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Mochi V1 Preview VAE.
        """
        return {
            "act_fn": "silu",
            "add_attention_block": [
                False,
                True,
                True,
                True,
                True
            ],
            "decoder_block_out_channels": [
                128,
                256,
                512,
                768
            ],
            "encoder_block_out_channels": [
                64,
                128,
                256,
                384
            ],
            "in_channels": 15,
            "latent_channels": 12,
            "latents_mean": [
                -0.06730895953510081,
                -0.038011381506090416,
                -0.07477820912866141,
                -0.05565264470995561,
                0.012767231469026969,
                -0.04703542746246419,
                0.043896967884726704,
                -0.09346305707025976,
                -0.09918314763016893,
                -0.008729793427399178,
                -0.011931556316503654,
                -0.0321993391887285
            ],
            "latents_std": [
                0.9263795028493863,
                0.9248894543193766,
                0.9393059390890617,
                0.959253732819592,
                0.8244560132752793,
                0.917259975397747,
                0.9294154431013696,
                1.3720942357788521,
                0.881393668867029,
                0.9168315692124348,
                0.9185249279345552,
                0.9274757570805041
            ],
            "layers_per_block": [
                3,
                3,
                4,
                6,
                3
            ],
            "out_channels": 3,
            "scaling_factor": 1.0,
            "spatial_expansions": [
                2,
                2,
                2
            ],
            "temporal_expansions": [
                1,
                2,
                3
            ]
        }
