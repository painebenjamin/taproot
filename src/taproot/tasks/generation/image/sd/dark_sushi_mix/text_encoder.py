from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionDarkSushiMixV225DTextEncoder"
]

class StableDiffusionDarkSushiMixV225DTextEncoder(CLIPViTLTextEncoder):
    """
    SDXL Counterfeit v2.5 Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-dark-sushi-mix-v2-25d-text-encoder.fp16.safetensors"
