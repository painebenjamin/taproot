from .base import StableDiffusionPretrainedLoRA

__all__ = [
    "AddDetailStableDiffusionPretrainedLoRA",
    "NoiseOffsetStableDiffusionPretrainedLoRA",
    "DPOStableDiffusionPretrainedLoRA"
]

class AddDetailStableDiffusionPretrainedLoRA(StableDiffusionPretrainedLoRA):
    name = "add-detail"
    url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-lora-add-detail.fp16.safetensors"
    author = "Lykon"
    author_url = "https://civitai.com/user/lykon"
    license = "OpenRAIL-M License with Addendum"
    license_attribution = False
    license_redistribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = True

class NoiseOffsetStableDiffusionPretrainedLoRA(StableDiffusionPretrainedLoRA):
    name = "noise-offset"
    url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-lora-noise-offset.fp16.safetensors"
    author = "epinikion"
    author_url = "https://civitai.com/user/epinikion"
    license = "OpenRAIL-M License with Addendum"
    license_attribution = False
    license_redistribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = False

class DPOStableDiffusionPretrainedLoRA(StableDiffusionPretrainedLoRA):
    name = "dpo"
    url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-lora-dpo.fp16.safetensors"
    author = "mhdang"
    author_url = "https://huggingface.co/mhdang"
    license = "OpenRAIL-M License with Addendum"
    license_attribution = False
    license_redistribution = True
    license_copy_left = False
    license_derivatives = True
    license_commercial = True
    license_hosting = True
