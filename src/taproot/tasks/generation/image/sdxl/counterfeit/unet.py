from ..pretrained import SDXLUNet

__all__ = ["SDXLCounterfeitV25UNet"]

class SDXLCounterfeitV25UNet(SDXLUNet):
    """
    SDXL DreamShaper Alpha V2 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-counterfeit-v2-5-unet.fp16.safetensors"
