from .base import StableDiffusionXLPretrainedLoRA

__all__ = [
    "NoiseOffsetStableDiffusionXLPretrainedLoRA",
    "DPOStableDiffusionXLPretrainedLoRA"
]

class NoiseOffsetStableDiffusionXLPretrainedLoRA(StableDiffusionXLPretrainedLoRA):
    name = "noise-offset"
    url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-lora-noise-offset.fp16.safetensors"
    author = "Stability AI"
    author_url = "https://huggingface.co/stabilityai"
    license = "OpenRAIL++-M License"
    license_attribution = False
    license_redistribution = True
    license_copy_left = False
    license_derivatives = True
    license_commercial = True
    license_hosting = True

class DPOStableDiffusionXLPretrainedLoRA(StableDiffusionXLPretrainedLoRA):
    name = "dpo"
    url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-lora-dpo.fp16.safetensors"
    author = "mhdang"
    author_url = "https://huggingface.co/mhdang"
    license = "OpenRAIL++-M License"
    license_attribution = False
    license_redistribution = True
    license_copy_left = False
    license_derivatives = True
    license_commercial = True
    license_hosting = True
