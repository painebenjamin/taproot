from .base import StableDiffusionPretrainedControlNet

__all__ = [
    "StableDiffusionPretrainedControlNetCannyEdge",
    "StableDiffusionPretrainedControlNetSoftEdge",
    "StableDiffusionPretrainedControlNetDepth",
    "StableDiffusionPretrainedControlNetHED",
    "StableDiffusionPretrainedControlNetAnimeLineArt",
    "StableDiffusionPretrainedControlNetLineArt",
    "StableDiffusionPretrainedControlNetMLSD",
    "StableDiffusionPretrainedControlNetNormal",
    "StableDiffusionPretrainedControlNetPose",
    "StableDiffusionPretrainedControlNetQRCode",
    "StableDiffusionPretrainedControlNetScribble",
    "StableDiffusionPretrainedControlNetSegment",
]

class StableDiffusionPretrainedControlNetCannyEdge(StableDiffusionPretrainedControlNet):
    name = "canny"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-controlnet-canny-edge.fp16.safetensors"

class StableDiffusionPretrainedControlNetSoftEdge(StableDiffusionPretrainedControlNet):
    name = "softedge"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-controlnet-soft-edge.fp16.safetensors"

class StableDiffusionPretrainedControlNetDepth(StableDiffusionPretrainedControlNet):
    name = "depth"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-controlnet-depth.fp16.safetensors"

class StableDiffusionPretrainedControlNetHED(StableDiffusionPretrainedControlNet):
    name = "hed"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-controlnet-hed.fp16.safetensors"

class StableDiffusionPretrainedControlNetAnimeLineArt(StableDiffusionPretrainedControlNet):
    name = "anime"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-controlnet-anime.fp16.safetensors"

class StableDiffusionPretrainedControlNetLineArt(StableDiffusionPretrainedControlNet):
    name = "lineart"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-controlnet-lineart.fp16.safetensors"

class StableDiffusionPretrainedControlNetMLSD(StableDiffusionPretrainedControlNet):
    name = "mlsd"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-controlnet-mlsd.fp16.safetensors"

class StableDiffusionPretrainedControlNetNormal(StableDiffusionPretrainedControlNet):
    name = "normal"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-controlnet-normal.fp16.safetensors"

class StableDiffusionPretrainedControlNetPose(StableDiffusionPretrainedControlNet):
    name = "pose"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-controlnet-pose.fp16.safetensors"

class StableDiffusionPretrainedControlNetQRCode(StableDiffusionPretrainedControlNet):
    name = "qr"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-controlnet-qr-code.fp16.safetensors"

class StableDiffusionPretrainedControlNetScribble(StableDiffusionPretrainedControlNet):
    name = "scribble"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-controlnet-scribble.fp16.safetensors"

class StableDiffusionPretrainedControlNetSegment(StableDiffusionPretrainedControlNet):
    name = "segment"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-controlnet-segment.fp16.safetensors"
