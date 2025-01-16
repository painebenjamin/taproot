from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionAbyssOrangeMixV3TextEncoder"
]

class StableDiffusionAbyssOrangeMixV3TextEncoder(CLIPViTLTextEncoder):
    """
    AbyssOrangeMix V3 Text Encoder
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-abyssorange-mix-v3-text-encoder.fp16.safetensors"
