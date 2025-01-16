from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionMeinaMixV12UNet"]

class StableDiffusionMeinaMixV12UNet(StableDiffusionUNet):
    """
    MeinaMix V12 Unet Model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-meinamix-v12-unet.fp16.safetensors"
