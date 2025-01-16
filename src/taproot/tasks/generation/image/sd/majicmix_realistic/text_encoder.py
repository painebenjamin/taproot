from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionMajicMixRealisticV7TextEncoder"
]

class StableDiffusionMajicMixRealisticV7TextEncoder(CLIPViTLTextEncoder):
    """
    MajicMix Realistic V7 Text Encoder
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-majicmix-realistic-v7-text-encoder.fp16.safetensors"
