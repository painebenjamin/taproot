from ..pretrained import SDXLUNet

__all__ = ["SDXLAnimagineV31UNet"]

class SDXLAnimagineV31UNet(SDXLUNet):
    """
    Animagine V3.1 UNet
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-animagine-v3-1-unet.fp16.safetensors"
