from .base import StableDiffusionPretrainedIPAdapter

__all__ = [
    "StableDiffusionIPAdapterBase",
    "StableDiffusionIPAdapterPlus",
    "StableDiffusionIPAdapterLight",
    "StableDiffusionIPAdapterPlusFace",
    "StableDiffusionIPAdapterFullFace"
]

class StableDiffusionIPAdapterBase(StableDiffusionPretrainedIPAdapter):
    recommended_scale = 1.1
    name = "base"
    url = "https://huggingface.co/benjamin-paine/taproot-common/image-generation-stable-diffusion-v1-5-ip-adapter.fp16.safetensors"

class StableDiffusionIPAdapterLight(StableDiffusionPretrainedIPAdapter):
    recommended_scale = 1.0
    name = "light"
    url = "https://huggingface.co/benjamin-paine/taproot-common/image-generation-stable-diffusion-v1-5-ip-adapter-light.fp16.safetensors"

class StableDiffusionIPAdapterPlus(StableDiffusionPretrainedIPAdapter):
    recommended_scale = 0.3
    name = "plus"
    url = "https://huggingface.co/benjamin-paine/taproot-common/image-generation-stable-diffusion-v1-5-ip-adapter-plus.fp16.safetensors"

class StableDiffusionIPAdapterPlusFace(StableDiffusionPretrainedIPAdapter):
    recommended_scale = 0.3
    name = "plus-face"
    url = "https://huggingface.co/benjamin-paine/taproot-common/image-generation-stable-diffusion-v1-5-ip-adapter-plus-face.fp16.safetensors"

class StableDiffusionIPAdapterFullFace(StableDiffusionPretrainedIPAdapter):
    recommended_scale = 0.3
    name = "full-face"
    url = "https://huggingface.co/benjamin-paine/taproot-common/image-generation-stable-diffusion-v1-5-ip-adapter-full-face.fp16.safetensors"
