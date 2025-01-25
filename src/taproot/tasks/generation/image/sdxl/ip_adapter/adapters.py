from .base import StableDiffusionXLPretrainedIPAdapter

__all__ = [
    "StableDiffusionXLIPAdapterBase",
    "StableDiffusionXLIPAdapterPlus",
    "StableDiffusionXLIPAdapterPlusFace",
]

class StableDiffusionXLIPAdapterBase(StableDiffusionXLPretrainedIPAdapter):
    recommended_scale = 0.3
    name = "base"
    url = "https://huggingface.co/benjamin-paine/taproot-common/image-generation-stable-diffusion-xl-ip-adapter.fp16.safetensors"

class StableDiffusionXLIPAdapterPlus(StableDiffusionXLPretrainedIPAdapter):
    recommended_scale = 0.2
    name = "plus"
    url = "https://huggingface.co/benjamin-paine/taproot-common/image-generation-stable-diffusion-xl-ip-adapter-plus.fp16.safetensors"

class StableDiffusionXLIPAdapterPlusFace(StableDiffusionXLPretrainedIPAdapter):
    recommended_scale = 0.3
    name = "plus-face"
    url = "https://huggingface.co/benjamin-paine/taproot-common/image-generation-stable-diffusion-xl-ip-adapter-plus-face.fp16.safetensors"
