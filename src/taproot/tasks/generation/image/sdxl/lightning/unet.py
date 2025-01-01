from ..pretrained import SDXLUNet

__all__ = [
    "SDXLLightningUNet8Step",
    "SDXLLightningUNet4Step",
    "SDXLLightningUNet2Step",
]

class SDXLLightningUNet8Step(SDXLUNet):
    """
    SDXL Lightning UNet model with 8 step training
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-lightning-unet-8-step.fp16.safetensors"

class SDXLLightningUNet4Step(SDXLUNet):
    """
    SDXL Lightning UNet model with 4 step training
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-lightning-unet-4-step.fp16.safetensors"

class SDXLLightningUNet2Step(SDXLUNet):
    """
    SDXL Lightning UNet model with 2 step training
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-lightning-unet-2-step.fp16.safetensors"
