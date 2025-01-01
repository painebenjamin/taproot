from ..pretrained import SDXLUNet

__all__ = ["SDXLAlbedoBaseV31UNet"]

class SDXLAlbedoBaseV31UNet(SDXLUNet):
    """
    SDXL AlbedoBaseV31 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-albedo-base-v3-1-unet.fp16.safetensors"
