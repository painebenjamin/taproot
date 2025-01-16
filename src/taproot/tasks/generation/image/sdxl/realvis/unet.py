from ..pretrained import SDXLUNet

__all__ = ["SDXLRealVisV50UNet"]

class SDXLRealVisV50UNet(SDXLUNet):
    """
    ReslVisXL V5.0 UNet
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-realvis-v5-0-unet.fp16.safetensors"
