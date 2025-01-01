from ..pretrained import SDXLUNet

__all__ = ["SDXLStoiqoNewRealityProUNet"]

class SDXLStoiqoNewRealityProUNet(SDXLUNet):
    """
    SDXL DreamShaper Alpha V2 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-stoiqo-newreality-pro-unet.fp16.safetensors"
