from taproot.pretrained import (
    CLIPViTLTextEncoderWithProjection,
    OpenCLIPViTGTextEncoder
)

__all__ = [
    "SDXLCounterfeitV25TextEncoderPrimary",
    "SDXLCounterfeitV25TextEncoderSecondary"
]

class SDXLCounterfeitV25TextEncoderPrimary(CLIPViTLTextEncoderWithProjection):
    """
    SDXL Counterfeit v2.5 Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-counterfeit-v2-5-text-encoder.fp16.safetensors"

class SDXLCounterfeitV25TextEncoderSecondary(OpenCLIPViTGTextEncoder):
    """
    SDXL Counterfeit v2.5 Secondary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-counterfeit-v2-5-text-encoder-2.fp16.safetensors"
