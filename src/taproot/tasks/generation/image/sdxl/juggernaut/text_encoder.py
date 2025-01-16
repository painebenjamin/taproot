from taproot.pretrained import (
    CLIPViTLTextEncoderWithProjection,
    OpenCLIPViTGTextEncoder
)

__all__ = [
    "SDXLJuggernautV11TextEncoderPrimary",
    "SDXLJuggernautV11TextEncoderSecondary"
]

class SDXLJuggernautV11TextEncoderPrimary(CLIPViTLTextEncoderWithProjection):
    """
    SDXL Juggernaut V11 Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-juggernaut-v11-text-encoder.fp16.safetensors"

class SDXLJuggernautV11TextEncoderSecondary(OpenCLIPViTGTextEncoder):
    """
    SDXL Juggernaut V11 Secondary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-juggernaut-v11-text-encoder-2.fp16.safetensors"
