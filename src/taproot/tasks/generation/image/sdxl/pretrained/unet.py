from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

__all__ = ["SDXLUNet"]

class SDXLUNet(PretrainedModelMixin):
    """
    The model for the Stable Diffusion XL UNet.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-base-unet.fp16.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[UNet2DConditionModel]:
        """
        Get the model class for the Stable Diffusion XL UNet.
        """
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
        return UNet2DConditionModel # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Stable Diffusion XL UNet.
        """
        return {
            "transformer_layers_per_block": [
                1,
                2,
                10
            ],
            "attention_head_dim": [
                5,
                10,
                20
            ],
            "block_out_channels": [
                320,
                640,
                1280
            ],
            "down_block_types": [
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D"
            ],
            "up_block_types": [
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D"
            ],
            "act_fn": "silu",
            "addition_embed_type": "text_time",
            "addition_embed_type_num_heads": 64,
            "addition_time_embed_dim": 256,
            "center_input_sample": False,
            "class_embed_type": None,
            "class_embeddings_concat": False,
            "conv_in_kernel": 3,
            "conv_out_kernel": 3,
            "cross_attention_dim": 2048,
            "cross_attention_norm": None,
            "downsample_padding": 1,
            "dual_cross_attention": False,
            "encoder_hid_dim": None,
            "encoder_hid_dim_type": None,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 4,
            "layers_per_block": 2,
            "mid_block_only_cross_attention": None,
            "mid_block_scale_factor": 1,
            "mid_block_type": "UNetMidBlock2DCrossAttn",
            "norm_eps": 0.00001,
            "norm_num_groups": 32,
            "num_attention_heads": None,
            "num_class_embeds": None,
            "only_cross_attention": False,
            "out_channels": 4,
            "projection_class_embeddings_input_dim": 2816,
            "resnet_out_scale_factor": 1,
            "resnet_skip_time_act": False,
            "resnet_time_scale_shift": "default",
            "sample_size": 128,
            "time_cond_proj_dim": None,
            "time_embedding_act_fn": None,
            "time_embedding_dim": None,
            "time_embedding_type": "positional",
            "timestep_post_act": None,
            "upcast_attention": None,
            "use_linear_projection": True
        }
