from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionChilloutMixNiTextEncoder"
]

class StableDiffusionChilloutMixNiTextEncoder(CLIPViTLTextEncoder):
    """
    Chillout Mix Ni Text Encoder
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-chillout-mix-ni-text-encoder.fp16.safetensors"
