from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionPerfectWorldV6UNet"]

class StableDiffusionPerfectWorldV6UNet(StableDiffusionUNet):
    """
    DreamShaper's UNet model, extracted.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-perfect-world-v6-unet.fp16.safetensors"
