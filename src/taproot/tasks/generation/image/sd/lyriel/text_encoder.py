from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionLyrielV16TextEncoder"
]

class StableDiffusionLyrielV16TextEncoder(CLIPViTLTextEncoder):
    """
    SDXL Counterfeit v2.5 Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-lyriel-v1-6-text-encoder.fp16.safetensors"
