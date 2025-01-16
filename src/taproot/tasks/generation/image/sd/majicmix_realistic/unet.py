from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionMajicMixRealisticV7UNet"]

class StableDiffusionMajicMixRealisticV7UNet(StableDiffusionUNet):
    """
    MajicMix Realistic V7 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-majicmix-realistic-v7-unet.fp16.safetensors"
