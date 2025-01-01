from ..pretrained import SDXLUNet

__all__ = ["SDXLCopaxTimeLessV13UNet"]

class SDXLCopaxTimeLessV13UNet(SDXLUNet):
    """
    SDXL DreamShaper Alpha V2 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-copax-timeless-v13-unet.fp16.safetensors"
