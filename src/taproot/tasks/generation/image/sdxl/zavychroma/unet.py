from ..pretrained import SDXLUNet

__all__ = ["SDXLZavyChromaV10UNet"]

class SDXLZavyChromaV10UNet(SDXLUNet):
    """
    ZavyChroma V10 UNet
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-zavychroma-v10-unet.fp16.safetensors"
