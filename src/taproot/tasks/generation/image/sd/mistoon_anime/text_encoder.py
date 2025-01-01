from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionMistoonAnimeV3TextEncoder"
]

class StableDiffusionMistoonAnimeV3TextEncoder(CLIPViTLTextEncoder):
    """
    SDXL Counterfeit v2.5 Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-mistoon-anime-v3-text-encoder.fp16.safetensors"
