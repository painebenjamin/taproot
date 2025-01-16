from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionDarkSushiMixV225DTextEncoder"
]

class StableDiffusionDarkSushiMixV225DTextEncoder(CLIPViTLTextEncoder):
    """
    DarkSushiMix V2 2.5D Text Encoder
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-dark-sushi-mix-v2-25d-text-encoder.fp16.safetensors"
