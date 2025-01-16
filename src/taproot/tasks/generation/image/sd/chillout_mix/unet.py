from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionChilloutMixNiUNet"]

class StableDiffusionChilloutMixNiUNet(StableDiffusionUNet):
    """
    Chillout Mix Ni UNet Models
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-chillout-mix-ni-unet.fp16.safetensors"
