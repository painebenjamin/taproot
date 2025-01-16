from ..pretrained.unet import StableDiffusionUNet

__all__ = [
    "StableDiffusionRealisticVisionV51UNet",
    "StableDiffusionRealisticVisionV60UNet"
]

class StableDiffusionRealisticVisionV51UNet(StableDiffusionUNet):
    """
    Realistic Vision V5.1 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-realistic-vision-v5-1-unet.fp16.safetensors"

class StableDiffusionRealisticVisionV60UNet(StableDiffusionUNet):
    """
    Realistic Vision V6.0 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-realistic-vision-v6-0-unet.fp16.safetensors"
