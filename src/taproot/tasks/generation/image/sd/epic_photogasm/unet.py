from ..pretrained.unet import StableDiffusionUNet

__all__ = ["StableDiffusionEpicPhotogasmUltimateFidelityUNet"]

class StableDiffusionEpicPhotogasmUltimateFidelityUNet(StableDiffusionUNet):
    """
    EpicPhotogasm Ultimate Fidelity UNet model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-epic-photogasm-ultimate-fidelity-unet.fp16.safetensors"
