from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionGhostMixV2TextEncoder"
]

class StableDiffusionGhostMixV2TextEncoder(CLIPViTLTextEncoder):
    """
    SDXL Counterfeit v2.5 Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-ghostmix-v2-text-encoder.fp16.safetensors"