from ..schnell import FluxSchnellInt8
from .transformer import (
    FluxSchnellSigmaVisionAlphaTransformerInt8,
)

__all__ = [
    "FluxSchnellSigmaVisionAlphaInt8",
]

class FluxSchnellSigmaVisionAlphaInt8(FluxSchnellInt8):
    """Global Task Metadata"""
    task = "image-generation"
    model = "flux-v1-schnell-sigma-vision-alpha-int8"
    do_true_cfg = True
    display_name = "Sigma Vision F1.S Alpha (Int8) Image Generation"
    pretrained_models = {
        **FluxSchnellInt8.pretrained_models,
        **{
            "transformer": FluxSchnellSigmaVisionAlphaTransformerInt8,
        },
    }
