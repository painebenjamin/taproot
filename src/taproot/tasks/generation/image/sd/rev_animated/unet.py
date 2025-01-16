from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionReVAnimatedV2UNet"]

class StableDiffusionReVAnimatedV2UNet(StableDiffusionUNet):
    """
    ReV Animation V2 UNet Model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-rev-animated-v2-unet.fp16.safetensors"
