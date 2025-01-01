from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo

__all__ = ["LTXVideoVAE"]

class LTXVideoVAE(PretrainedModelMixin):
    """
    The model for the LTX Video VAE.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-ltx-vae.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[AutoencoderKLLTXVideo]:
        """
        Get the model class for the LTX Video VAE.
        """
        from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
        return AutoencoderKLLTXVideo # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the LTX Video VAE.
        """
        return {
            "block_out_channels": [128, 256, 512, 512],
            "decoder_block_out_channels": [256, 512, 1024],
            "decoder_causal": False,
            "decoder_inject_noise": [True, True, True, False],
            "decoder_layers_per_block": [5, 6, 7, 8],
            "decoder_spatio_temporal_scaling": [True, True, True],
            "encoder_causal": True,
            "in_channels": 3,
            "latent_channels": 128,
            "layers_per_block": [4,3,3,3,4],
            "out_channels": 3,
            "patch_size": 4,
            "patch_size_t": 1,
            "resnet_norm_eps": 1e-06,
            "scaling_factor": 1.0,
            "spatio_temporal_scaling": [True, True, True, False],
            "timestep_conditioning": True,
            "upsample_factor": [2, 2, 2],
            "upsample_residual": [True, True, True],
        }
