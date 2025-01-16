from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionToonYouBetaV6TextEncoder"
]

class StableDiffusionToonYouBetaV6TextEncoder(CLIPViTLTextEncoder):
    """
    ToonYou Beta V6 Text Encoder
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-toonyou-beta-v6-text-encoder.fp16.safetensors"
