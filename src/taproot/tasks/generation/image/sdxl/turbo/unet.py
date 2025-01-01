from typing import Any, Dict, Optional

from ..pretrained import SDXLUNet

__all__ = ["SDXLTurboUNet"]

class SDXLTurboUNet(SDXLUNet):
    """
    SDXL Turbo UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-xl-turbo-unet.fp16.safetensors"
    use_compile: bool = True

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the model
        We reduce the sample size from 128 to 64
        """
        config = super().get_default_config()
        assert isinstance(config, dict), "base config must be a dictionary"
        config["sample_size"] = 64
        return config
