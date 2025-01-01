from taproot.pretrained import (
    CLIPViTLTextEncoderWithProjection,
    OpenCLIPViTGTextEncoder
)

__all__ = [
    "SDXLUnstableDiffusersNihilmaniaTextEncoderPrimary",
    "SDXLUnstableDiffusersNihilmaniaTextEncoderSecondary"
]

class SDXLUnstableDiffusersNihilmaniaTextEncoderPrimary(CLIPViTLTextEncoderWithProjection):
    """
    SDXL Counterfeit v2.5 Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-unstable-diffusers-nihilmania-text-encoder.fp16.safetensors"

class SDXLUnstableDiffusersNihilmaniaTextEncoderSecondary(OpenCLIPViTGTextEncoder):
    """
    SDXL Counterfeit v2.5 Secondary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-unstable-diffusers-nihilmania-text-encoder-2.fp16.safetensors"
