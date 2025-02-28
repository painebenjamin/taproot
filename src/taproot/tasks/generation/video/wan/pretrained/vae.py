from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING

from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from ..model import WanVideoVAE

__all__ = ["PretrainedWanVAE"]

class PretrainedWanVAE(PretrainedModelMixin):
    """
    WanVideoVAE model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-wan-v2-1-vae.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "dim": 96,
            "z_dim": 16,
            "dim_mult": [1,2,4,4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temporal_downsample": [False, True, True],
            "dropout": 0.0,
            "mean": [
                -0.7571, -0.7089, -0.9113, 0.1075,
                -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632,
                -0.1922, -0.9497, 0.2503, -0.2921,
            ],
            "std": [
                2.8184, 1.4541, 2.3275, 2.6558,
                1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579,
                1.6382, 1.1253, 2.8251, 1.9160,
            ],
            "stride": (4, 8, 8),
        }

    @classmethod
    def get_model_class(cls) -> Type[WanVideoVAE]:
        """
        Returns the model class.
        """
        from ..model import WanVideoVAE
        return WanVideoVAE
