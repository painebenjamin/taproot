from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

__all__ = ["StableDiffusionUNet"]

class StableDiffusionUNet(PretrainedModelMixin):
    """
    The model for the Stable Diffusion UNet.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-unet.fp16.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[UNet2DConditionModel]:
        """
        Get the model class for the Stable Diffusion UNet.
        """
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
        return UNet2DConditionModel # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Stable Diffusion UNet.
        """
        return {
            "act_fn": "silu",
            "attention_head_dim": 8,
            "block_out_channels": [
                320,
                640,
                1280,
                1280
            ],
            "center_input_sample": False,
            "cross_attention_dim": 768,
            "down_block_types": [
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D"
            ],
            "downsample_padding": 1,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 4,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "norm_eps": 0.00001,
            "norm_num_groups": 32,
            "out_channels": 4,
            "sample_size": 64,
            "up_block_types": [
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D"
            ]
        }
