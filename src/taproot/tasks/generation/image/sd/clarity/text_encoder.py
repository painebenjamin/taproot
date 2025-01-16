from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionClarityV3TextEncoder"
]

class StableDiffusionClarityV3TextEncoder(CLIPViTLTextEncoder):
    """
    Clarity V3 Text Encoder
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-clarity-v3-text-encoder.fp16.safetensors"
