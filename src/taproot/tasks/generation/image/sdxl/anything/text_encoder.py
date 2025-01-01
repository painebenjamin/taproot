from taproot.pretrained import (
    CLIPViTLTextEncoderWithProjection,
    OpenCLIPViTGTextEncoder
)

__all__ = [
    "SDXLAnythingTextEncoderPrimary",
    "SDXLAnythingTextEncoderSecondary"
]

class SDXLAnythingTextEncoderPrimary(CLIPViTLTextEncoderWithProjection):
    """
    SDXL Counterfeit v2.5 Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-anything-text-encoder.fp16.safetensors"

class SDXLAnythingTextEncoderSecondary(OpenCLIPViTGTextEncoder):
    """
    SDXL Counterfeit v2.5 Secondary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-anything-text-encoder-2.fp16.safetensors"
