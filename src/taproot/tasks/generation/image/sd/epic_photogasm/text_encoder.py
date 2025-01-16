from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionEpicPhotogasmUltimateFidelityTextEncoder"
]

class StableDiffusionEpicPhotogasmUltimateFidelityTextEncoder(CLIPViTLTextEncoder):
    """
    Epic Photogasm Ultimate Fidelity Text Encoder
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-epic-photogasm-ultimate-fidelity-text-encoder.fp16.safetensors"
