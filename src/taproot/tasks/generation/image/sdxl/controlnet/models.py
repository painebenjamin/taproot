from .base import StableDiffusionXLPretrainedControlNet

__all__ = [
    "StableDiffusionXLPretrainedControlNetCannyEdge",
    "StableDiffusionXLPretrainedControlNetDepth",
    "StableDiffusionXLPretrainedControlNetPose",
    "StableDiffusionXLPretrainedControlNetQRCode",
    "StableDiffusionXLPretrainedControlNetScribble",
]

class StableDiffusionXLPretrainedControlNetCannyEdge(StableDiffusionXLPretrainedControlNet):
    name = "canny"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-controlnet-canny-edge.fp16.safetensors"

class StableDiffusionXLPretrainedControlNetDepth(StableDiffusionXLPretrainedControlNet):
    name = "depth"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-controlnet-depth.fp16.safetensors"

class StableDiffusionXLPretrainedControlNetPose(StableDiffusionXLPretrainedControlNet):
    name = "pose"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-controlnet-pose.fp16.safetensors"

class StableDiffusionXLPretrainedControlNetQRCode(StableDiffusionXLPretrainedControlNet):
    name = "qr"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-controlnet-qr-code.fp16.safetensors"

class StableDiffusionXLPretrainedControlNetScribble(StableDiffusionXLPretrainedControlNet):
    name = "scribble"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-controlnet-scribble.fp16.safetensors"
