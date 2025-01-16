from ..pretrained import SDXLUNet

__all__ = ["SDXLAnythingUNet"]

class SDXLAnythingUNet(SDXLUNet):
    """
    Anything XL Unet
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-anything-unet.fp16.safetensors"
