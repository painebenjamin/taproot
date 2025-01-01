from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionGhostMixV2UNet"]

class StableDiffusionGhostMixV2UNet(StableDiffusionUNet):
    """
    DreamShaper's UNet model, extracted.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-ghostmix-v2-unet.fp16.safetensors"
