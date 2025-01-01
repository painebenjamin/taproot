from ..pretrained import SDXLUNet

__all__ = ["SDXLUnstableDiffusersNihilmaniaUNet"]

class SDXLUnstableDiffusersNihilmaniaUNet(SDXLUNet):
    """
    SDXL DreamShaper Alpha V2 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-unstable-diffusers-nihilmania-unet.fp16.safetensors"
