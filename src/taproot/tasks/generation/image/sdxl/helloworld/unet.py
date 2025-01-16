from ..pretrained import SDXLUNet

__all__ = ["SDXLHelloWorldV7UNet"]

class SDXLHelloWorldV7UNet(SDXLUNet):
    """
    HelloWorld V7 UNet
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-hello-world-v7-unet.fp16.safetensors"
