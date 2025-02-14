# Adapted from https://github.com/Zyphra/Zonos
import torch
import torch.nn as nn

from typing import Optional
from mamba_ssm.models.mixer_seq_simple import create_block # type: ignore[import-untyped]
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn # type: ignore[import-untyped]
from mamba_ssm.utils.generation import InferenceParams # type: ignore[import-untyped]

from .config import BackboneConfig

__all__ = ["ZonosBackbone"]

class ZonosBackbone(nn.Module):
    """
    Zonos backbone model
    """
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model=config.d_model,
                    d_intermediate=(
                        config.d_intermediate
                        if (i not in config.attn_layer_idx)
                        else config.attn_mlp_d_intermediate
                    ),
                    ssm_cfg=config.ssm_cfg,
                    layer_idx=i,
                    attn_layer_idx=config.attn_layer_idx,
                    attn_cfg=config.attn_cfg,
                    norm_epsilon=config.norm_epsilon,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=True,
                    rms_norm=config.rms_norm,
                )
                for i in range(config.n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(
            config.d_model,
            eps=config.norm_epsilon
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params: Optional[InferenceParams]=None
    ) -> torch.Tensor:
        """
        Forward pass of the model
        """
        residual: Optional[torch.Tensor] = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params)

        return layer_norm_fn( # type: ignore[no-any-return]
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            residual,
            eps=self.norm_f.eps,
            residual_in_fp32=self.config.residual_in_fp32,
            is_rms_norm=self.config.rms_norm,
        )
