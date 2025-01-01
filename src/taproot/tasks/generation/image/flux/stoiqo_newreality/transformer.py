from ..pretrained.transformer import (
    FluxDevTransformer,
    FluxDevTransformerInt8,
    FluxDevTransformerNF4
)

__all__ = [
    "FluxDevStoiqoNewRealityAlphaV2TransformerFP8",
    "FluxDevStoiqoNewRealityAlphaV2TransformerInt8",
    "FluxDevStoiqoNewRealityAlphaV2TransformerNF4"
]

class FluxDevStoiqoNewRealityAlphaV2TransformerFP8(FluxDevTransformer):
    """
    Stoiqo New Reality for FLUX.dev Alpha v2 Transformer model in FP8.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-dev-stoiqo-newreality-alpha-v2-transformer.fp8-e4m3-fn.safetensors"
    dtype = "float8_e4m3fn"

class FluxDevStoiqoNewRealityAlphaV2TransformerInt8(FluxDevTransformerInt8):
    """
    Stoiqo New Reality for FLUX.dev Alpha v2 Transformer model.
    Pre-quantized to INT8 with bitsandbytes.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-dev-stoiqo-newreality-alpha-v2-transformer.int8.fp16.safetensors"

class FluxDevStoiqoNewRealityAlphaV2TransformerNF4(FluxDevTransformerNF4):
    """
    Stoiqo New Reality for FLUX.dev Alpha v2 Transformer model
    Pre-quantized to NF4 with bitsandbytes.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-dev-stoiqo-newreality-alpha-v2-transformer.nf4.fp16.safetensors"
