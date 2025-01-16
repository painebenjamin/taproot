from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionPerfectWorldV6TextEncoder"
]

class StableDiffusionPerfectWorldV6TextEncoder(CLIPViTLTextEncoder):
    """
    PerfectWorld V6 Text Encoder
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-perfect-world-v6-text-encoder.fp16.safetensors"
