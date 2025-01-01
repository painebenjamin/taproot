from ..pretrained import SDXLUNet

__all__ = ["SDXLRealVisV50UNet"]

class SDXLRealVisV50UNet(SDXLUNet):
    """
    SDXL DreamShaper Alpha V2 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-realvis-v5-0-unet.fp16.safetensors"
