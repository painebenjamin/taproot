from ..pretrained import SDXLUNet

__all__ = ["SDXLJuggernautV11UNet"]

class SDXLJuggernautV11UNet(SDXLUNet):
    """
    SDXL DreamShaper Alpha V2 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-juggernaut-v11-unet.fp16.safetensors"
