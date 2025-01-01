from taproot.pretrained import (
    CLIPViTLTextEncoderWithProjection,
    OpenCLIPViTGTextEncoder
)

__all__ = [
    "SDXLDreamShaperAlphaV2TextEncoderPrimary",
    "SDXLDreamShaperAlphaV2TextEncoderSecondary"
]

class SDXLDreamShaperAlphaV2TextEncoderPrimary(CLIPViTLTextEncoderWithProjection):
    """
    SDXL Counterfeit v2.5 Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-dreamshaper-alpha-v2-text-encoder.fp16.safetensors"

class SDXLDreamShaperAlphaV2TextEncoderSecondary(OpenCLIPViTGTextEncoder):
    """
    SDXL Counterfeit v2.5 Secondary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-dreamshaper-alpha-v2-text-encoder-2.fp16.safetensors"