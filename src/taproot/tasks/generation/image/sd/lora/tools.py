from .base import StableDiffusionHostedLoRA

__all__ = [
    "AddDetailStableDiffusionHostedLoRA",
    "NoiseOffsetStableDiffusionHostedLoRA",
]

class AddDetailStableDiffusionHostedLoRA(StableDiffusionHostedLoRA):
    name = "add-detail"
    url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-lora-add-detail.fp16.safetensors"
    author = "Lykon"
    author_url = "https://civitai.com/user/lykon"
    license = "OpenRAIL-M License with Restrictions"
    license_attribution = False
    license_redistribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = True

class NoiseOffsetStableDiffusionHostedLoRA(StableDiffusionHostedLoRA):
    name = "noise-offset"
    url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-lora-noise-offset.fp16.safetensors"
    author = "epinikion"
    author_url = "https://civitai.com/user/epinikion"
    license = "OpenRAIL-M License with Restrictions"
    license_attribution = False
    license_redistribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = False
