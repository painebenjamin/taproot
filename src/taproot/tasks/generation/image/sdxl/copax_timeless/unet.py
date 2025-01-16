from ..pretrained import SDXLUNet

__all__ = ["SDXLCopaxTimeLessV13UNet"]

class SDXLCopaxTimeLessV13UNet(SDXLUNet):
    """
    Copax TimeLess V13 UNet
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-copax-timeless-v13-unet.fp16.safetensors"
