from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionReVAnimatedV2TextEncoder"
]

class StableDiffusionReVAnimatedV2TextEncoder(CLIPViTLTextEncoder):
    """
    ReV Animated V2 Text Encoder
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-rev-animated-v2-text-encoder.fp16.safetensors"
