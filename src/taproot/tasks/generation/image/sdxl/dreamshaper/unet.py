from ..pretrained import SDXLUNet

__all__ = ["SDXLDreamShaperAlphaV2UNet"]

class SDXLDreamShaperAlphaV2UNet(SDXLUNet):
    """
    DreamShaper Alpha V2 UNet
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-dreamshaper-alpha-v2-unet.fp16.safetensors"
