from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionEpicRealismV5UNet"]

class StableDiffusionEpicRealismV5UNet(StableDiffusionUNet):
    """
    EpicRealism V5 UNet Model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-epicrealism-v5-unet.fp16.safetensors"
