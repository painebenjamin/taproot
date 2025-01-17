from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionSerenityV21UNet"]

class StableDiffusionSerenityV21UNet(StableDiffusionUNet):
    """
    Serenity V2.1 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-serenity-v2-1-unet.fp16.safetensors"
