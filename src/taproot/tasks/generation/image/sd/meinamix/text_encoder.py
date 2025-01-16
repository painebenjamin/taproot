from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionMeinaMixV12TextEncoder"
]

class StableDiffusionMeinaMixV12TextEncoder(CLIPViTLTextEncoder):
    """
    MeinaMix V12 Text Encoder
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-meinamix-v12-text-encoder.fp16.safetensors"
