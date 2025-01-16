from ..pretrained import SDXLUNet

__all__ = ["SDXLUnstableDiffusersNihilmaniaUNet"]

class SDXLUnstableDiffusersNihilmaniaUNet(SDXLUNet):
    """
    Unstable Diffussers Nihilmania UNet
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-unstable-diffusers-nihilmania-unet.fp16.safetensors"
