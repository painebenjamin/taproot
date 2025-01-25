from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin, NamedDiscoveryMixin

if TYPE_CHECKING:
    from diffusers.models.controlnets.controlnet import ControlNetModel

__all__ = ["StableDiffusionXLPretrainedControlNet"]

class StableDiffusionXLPretrainedControlNet(PretrainedModelMixin, NamedDiscoveryMixin):
    """
    A base class for ControlNets for Stable Diffusion XL
    """
    @classmethod
    def get_model_class(cls) -> Type[ControlNetModel]:
        """
        Returns the model class that this control net is associated with.
        """
        from diffusers.models.controlnets.controlnet import ControlNetModel
        return ControlNetModel # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for this control net.
        """
        return {
            "act_fn": "silu",
            "addition_embed_type": "text_time",
            "addition_embed_type_num_heads": 64,
            "addition_time_embed_dim": 256,
            "attention_head_dim": [5, 10, 20],
            "block_out_channels": [320, 640, 1280],
            "class_embed_type": None,
            "conditioning_channels": 3,
            "conditioning_embedding_out_channels": [16, 32, 96, 256],
            "controlnet_conditioning_channel_order": "rgb",
            "cross_attention_dim": 2048,
            "down_block_types": [
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D"
            ],
            "downsample_padding": 1,
            "encoder_hid_dim": None,
            "encoder_hid_dim_type": None,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "global_pool_conditions": False,
            "in_channels": 4,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "norm_eps": 0.00001,
            "norm_num_groups": 32,
            "num_attention_heads": None,
            "num_class_embeds": None,
            "only_cross_attention": False,
            "projection_class_embeddings_input_dim": 2816,
            "resnet_time_scale_shift": "default",
            "transformer_layers_per_block": [1, 2, 10],
            "upcast_attention": None,
            "use_linear_projection": True
        }
