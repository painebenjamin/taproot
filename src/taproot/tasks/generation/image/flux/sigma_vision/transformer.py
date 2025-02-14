from ..pretrained.transformer import (
    FluxSchnellTransformer,
    FluxSchnellTransformerInt8,
)

__all__ = [
    "FluxSchnellSigmaVisionAlphaTransformerFP8",
    "FluxSchnellSigmaVisionAlphaTransformerInt8",
]

class FluxSchnellSigmaVisionAlphaTransformerFP8(FluxSchnellTransformer):
    """
    SigmaVision Alpha for FLUX.dev Transformer model in FP8.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-dev-sigma-vision-alpha-transformer.fp8-e4m3-fn.safetensors"
    dtype = "float8_e4m3fn"

class FluxSchnellSigmaVisionAlphaTransformerInt8(FluxSchnellTransformerInt8):
    """
    SigmaVision Alpha for FLUX.dev Transformer model.
    Pre-quantized to INT8 with bitsandbytes.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-dev-sigma-vision-alpha-transformer.int8.fp16.safetensors"
