from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionToonYouBetaV6UNet"]

class StableDiffusionToonYouBetaV6UNet(StableDiffusionUNet):
    """
    DreamShaper's UNet model, extracted.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-toonyou-beta-v6-unet.fp16.safetensors"
