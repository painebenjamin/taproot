from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionClarityV3UNet"]

class StableDiffusionClarityV3UNet(StableDiffusionUNet):
    """
    Clarity V3 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-clarity-v3-unet.fp16.safetensors"
