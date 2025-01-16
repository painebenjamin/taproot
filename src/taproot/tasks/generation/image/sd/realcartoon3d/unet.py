from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionRealCartoon3DV17UNet"]

class StableDiffusionRealCartoon3DV17UNet(StableDiffusionUNet):
    """
    RealCartoon 3D V17 UNet
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-realcartoon3d-v17-unet.fp16.safetensors"
