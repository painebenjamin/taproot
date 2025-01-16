from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionPerfectWorldV6UNet"]

class StableDiffusionPerfectWorldV6UNet(StableDiffusionUNet):
    """
    Perfect World V6 UNet model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-perfect-world-v6-unet.fp16.safetensors"
