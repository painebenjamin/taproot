from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionFilmGirlUltraUNet"]

class StableDiffusionFilmGirlUltraUNet(StableDiffusionUNet):
    """
    FilmGirls's UNet model, extracted.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-filmgirl-ultra-unet.fp16.safetensors"
