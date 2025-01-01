from ..pretrained import SDXLUNet

__all__ = ["SDXLAnythingUNet"]

class SDXLAnythingUNet(SDXLUNet):
    """
    SDXL DreamShaper Alpha V2 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-anything-unet.fp16.safetensors"
