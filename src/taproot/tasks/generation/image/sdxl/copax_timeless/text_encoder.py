from taproot.pretrained import (
    CLIPViTLTextEncoderWithProjection,
    OpenCLIPViTGTextEncoder
)

__all__ = [
    "SDXLCopaxTimeLessV13TextEncoderPrimary",
    "SDXLCopaxTimeLessV13TextEncoderSecondary"
]

class SDXLCopaxTimeLessV13TextEncoderPrimary(CLIPViTLTextEncoderWithProjection):
    """
    SDXL Counterfeit v2.5 Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-copax-timeless-v13-text-encoder.fp16.safetensors"

class SDXLCopaxTimeLessV13TextEncoderSecondary(OpenCLIPViTGTextEncoder):
    """
    SDXL Counterfeit v2.5 Secondary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-copax-timeless-v13-text-encoder-2.fp16.safetensors"
