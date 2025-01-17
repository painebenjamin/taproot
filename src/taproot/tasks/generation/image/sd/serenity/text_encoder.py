from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionSerenityV21TextEncoder"
]

class StableDiffusionSerenityV21TextEncoder(CLIPViTLTextEncoder):
    """
    Serenity V21 Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-serenity-v2-1-text-encoder.fp16.safetensors"
