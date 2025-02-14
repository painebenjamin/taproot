# Adapted from https://github.com/Zyphra/Zonos
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, List, Dict, Any

__all__ = ["BackboneConfig", "PrefixConditionerConfig", "ZonosConfig"]

@dataclass
class BackboneConfig:
    d_model: int = 1024
    d_intermediate: int = 0
    attn_mlp_d_intermediate: int = 0
    n_layer: int = 16
    ssm_cfg: Dict[str, Any] = field(default_factory=dict)
    attn_layer_idx: List[int] = field(default_factory=list)
    attn_cfg: Dict[str, Any] = field(default_factory=dict)
    rms_norm: bool = False
    residual_in_fp32: bool = False
    norm_epsilon: float = 1e-5

@dataclass
class PrefixConditionerConfig:
    conditioners: List[Dict[str, Any]]
    projection: Literal["none", "linear", "mlp"]

@dataclass
class ZonosConfig:
    backbone: BackboneConfig
    prefix_conditioner: PrefixConditionerConfig
    eos_token_id: int = 1024
    masked_token_id: int = 1025
    n_codebooks: int = 9

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ZonosConfig:
        """
        Create a ZonosConfig object from a dictionary.
        """
        d = d.copy()
        backbone_config = BackboneConfig(**d.pop("backbone"))
        prefix_conditioner_config = PrefixConditionerConfig(**d.pop("prefix_conditioner"))
        config = cls(backbone_config, prefix_conditioner_config, **d)
        return config
