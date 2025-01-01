from ..pretrained import SDXLUNet

__all__ = ["SDXLDreamShaperAlphaV2UNet"]

class SDXLDreamShaperAlphaV2UNet(SDXLUNet):
    """
    SDXL DreamShaper Alpha V2 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-dreamshaper-alpha-v2-unet.fp16.safetensors"
