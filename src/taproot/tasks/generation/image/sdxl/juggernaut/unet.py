from ..pretrained import SDXLUNet

__all__ = ["SDXLJuggernautV11UNet"]

class SDXLJuggernautV11UNet(SDXLUNet):
    """
    Juggernaut V11 UNet
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-juggernaut-v11-unet.fp16.safetensors"
