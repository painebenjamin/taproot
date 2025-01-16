from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionAbyssOrangeMixV3UNet"]

class StableDiffusionAbyssOrangeMixV3UNet(StableDiffusionUNet):
    """
    AbyssOrangeMix V3 UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-abyssorange-mix-v3-unet.fp16.safetensors"
