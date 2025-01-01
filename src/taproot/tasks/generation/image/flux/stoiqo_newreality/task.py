from ..dev import FluxDevInt8, FluxDevNF4
from .transformer import (
    FluxDevStoiqoNewRealityAlphaV2TransformerInt8,
    FluxDevStoiqoNewRealityAlphaV2TransformerNF4
)

__all__ = [
    "FluxDevStoiqoNewRealityAlphaV2Int8",
    "FluxDevStoiqoNewRealityAlphaV2NF4"
]

class FluxDevStoiqoNewRealityAlphaV2Int8(FluxDevInt8):
    """Global Task Metadata"""
    task = "image-generation"
    model = "flux-v1-dev-stoiqo-newreality-alpha-v2-int8"
    display_name = "Stoiqo NewReality F1.D Alpha V2 (Int8) Image Generation"
    pretrained_models = {
        **FluxDevInt8.pretrained_models,
        **{
            "transformer": FluxDevStoiqoNewRealityAlphaV2TransformerInt8,
        },
    }

class FluxDevStoiqoNewRealityAlphaV2NF4(FluxDevNF4):
    """Global Task Metadata"""
    task = "image-generation"
    model = "flux-v1-dev-stoiqo-newreality-alpha-v2-nf4"
    display_name = "Stoiqo NewReality F1.D Alpha V2 (NF4) Image Generation"
    pretrained_models = {
        **FluxDevNF4.pretrained_models,
        **{
            "transformer": FluxDevStoiqoNewRealityAlphaV2TransformerNF4,
        },
    }
