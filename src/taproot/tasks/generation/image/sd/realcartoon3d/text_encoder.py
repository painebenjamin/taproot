from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionRealCartoon3DV17TextEncoder"
]

class StableDiffusionRealCartoon3DV17TextEncoder(CLIPViTLTextEncoder):
    """
    SDXL Counterfeit v2.5 Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-realcartoon3d-v17-text-encoder.fp16.safetensors"
