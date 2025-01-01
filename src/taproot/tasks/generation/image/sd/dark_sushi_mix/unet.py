from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionDarkSushiMixV225DUNet"]

class StableDiffusionDarkSushiMixV225DUNet(StableDiffusionUNet):
    """
    DreamShaper's UNet model, extracted.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-dark-sushi-mix-v2-25d-unet.fp16.safetensors"
