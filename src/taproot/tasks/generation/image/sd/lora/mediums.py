from .base import StableDiffusionPretrainedLoRA

__all__ = [
    "PolaroidStableDiffusionPretrainedLoRA",
]

class PolaroidStableDiffusionPretrainedLoRA(StableDiffusionPretrainedLoRA):
    name = "polaroid"
    url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-lora-polaroid.fp16.safetensors"
    author = "LEOSAM"
    author_url = "https://civitai.com/user/LEOSAM"
    license = "OpenRAIL-M License with Addendum"
    license_attribution = False
    license_redistribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = False
