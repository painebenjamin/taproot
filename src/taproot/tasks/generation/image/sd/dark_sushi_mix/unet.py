from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionDarkSushiMixV225DUNet"]

class StableDiffusionDarkSushiMixV225DUNet(StableDiffusionUNet):
    """
    DarkSushi Mix V2 2.5D UNet Model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-dark-sushi-mix-v2-25d-unet.fp16.safetensors"
