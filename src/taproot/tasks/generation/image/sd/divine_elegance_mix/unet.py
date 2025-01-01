from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionDivineEleganceMixV10UNet"]

class StableDiffusionDivineEleganceMixV10UNet(StableDiffusionUNet):
    """
    DreamShaper's UNet model, extracted.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-divine-elegance-mix-v10-unet.fp16.safetensors"
