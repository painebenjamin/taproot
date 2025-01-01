from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionMistoonAnimeV3UNet"]

class StableDiffusionMistoonAnimeV3UNet(StableDiffusionUNet):
    """
    DreamShaper's UNet model, extracted.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-mistoon-anime-v3-unet.fp16.safetensors"
