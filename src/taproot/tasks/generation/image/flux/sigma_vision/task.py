from ..schnell import FluxSchnellInt8, FluxSchnellNF4
from .transformer import (
    FluxSchnellSigmaVisionAlphaTransformerInt8,
    FluxSchnellSigmaVisionAlphaTransformerNF4
)

__all__ = [
    "FluxSchnellSigmaVisionAlphaInt8",
    "FluxSchnellSigmaVisionAlphaNF4"
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

class FluxSchnellSigmaVisionAlphaNF4(FluxSchnellNF4):
    """Global Task Metadata"""
    task = "image-generation"
    model = "flux-v1-schnell-sigma-vision-alpha-nf4"
    do_true_cfg = True
    display_name = "Sigma Vision F1.S Alpha (NF4) Image Generation"
    pretrained_models = {
        **FluxSchnellNF4.pretrained_models,
        **{
            "transformer": FluxSchnellSigmaVisionAlphaTransformerNF4,
        },
    }
