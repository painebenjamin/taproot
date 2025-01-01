from taproot.pretrained import (
    CLIPViTLTextEncoderWithProjection,
    OpenCLIPViTGTextEncoder
)

__all__ = [
    "SDXLAlbedoBaseV31TextEncoderPrimary",
    "SDXLAlbedoBaseV31TextEncoderSecondary"
]

class SDXLAlbedoBaseV31TextEncoderPrimary(CLIPViTLTextEncoderWithProjection):
    """
    SDXL Counterfeit v2.5 Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-albedo-base-v3-1-text-encoder.fp16.safetensors"

class SDXLAlbedoBaseV31TextEncoderSecondary(OpenCLIPViTGTextEncoder):
    """
    SDXL Counterfeit v2.5 Secondary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-albedo-base-v3-1-text-encoder-2.fp16.safetensors"
