from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionChilloutMixNiUNet"]

class StableDiffusionChilloutMixNiUNet(StableDiffusionUNet):
    """
    DreamShaper's UNet model, extracted.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-chillout-mix-ni-unet.fp16.safetensors"
