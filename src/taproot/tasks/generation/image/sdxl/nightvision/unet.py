from ..pretrained import SDXLUNet

__all__ = ["SDXLNightVisionV9UNet"]

class SDXLNightVisionV9UNet(SDXLUNet):
    """
    NightVision V9 UNet
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-nightvision-v9-unet.fp16.safetensors"
