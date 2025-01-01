from ..pretrained.unet import StableDiffusionUNet

__all__ = [
    "StableDiffusionRealisticVisionV51UNet",
    "StableDiffusionRealisticVisionV60UNet"
]

class StableDiffusionRealisticVisionV51UNet(StableDiffusionUNet):
    """
    DreamShaper's UNet model, extracted.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-realistic-vision-v5-1-unet.fp16.safetensors"

class StableDiffusionRealisticVisionV60UNet(StableDiffusionUNet):
    """
    DreamShaper's UNet model, extracted.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-realistic-vision-v6-0-unet.fp16.safetensors"
