from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionDreamShaperV8UNet"]

class StableDiffusionDreamShaperV8UNet(StableDiffusionUNet):
    """
    DreamShaper's UNet model, extracted.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-dreamshaper-v8-unet.fp16.safetensors"
