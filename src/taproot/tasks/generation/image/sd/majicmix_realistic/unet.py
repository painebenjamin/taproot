from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionMajicMixRealisticV7UNet"]

class StableDiffusionMajicMixRealisticV7UNet(StableDiffusionUNet):
    """
    DreamShaper's UNet model, extracted.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-majicmix-realistic-v7-unet.fp16.safetensors"
