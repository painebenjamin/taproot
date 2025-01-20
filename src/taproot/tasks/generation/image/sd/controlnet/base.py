from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin, NamedDiscoveryMixin

if TYPE_CHECKING:
    from diffusers.models.controlnets.controlnet import ControlNetModel

__all__ = ["StableDiffusionPretrainedControlNet"]

class StableDiffusionPretrainedControlNet(PretrainedModelMixin, NamedDiscoveryMixin):
    """
    A base class for ControlNets for Stable Diffusion 1.5.
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
            "attention_head_dim": 8,
            "block_out_channels": [320, 640, 1280, 1280],
            "class_embed_type": None,
            "conditioning_embedding_out_channels": [16, 32, 96, 256],
            "controlnet_conditioning_channel_order": "rgb",
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
            "norm_eps": 1e-05,
            "norm_num_groups": 32,
            "num_class_embeds": None,
            "only_cross_attention": False,
            "projection_class_embeddings_input_dim": None,
            "resnet_time_scale_shift": "default",
            "upcast_attention": False,
            "use_linear_projection": False
        }
