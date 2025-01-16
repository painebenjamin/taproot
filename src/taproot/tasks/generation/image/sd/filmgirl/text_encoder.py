from taproot.pretrained import CLIPViTLTextEncoder

__all__ = [
    "StableDiffusionFilmGirlUltraTextEncoder"
]

class StableDiffusionFilmGirlUltraTextEncoder(CLIPViTLTextEncoder):
    """
    FilmGirl Ultra Primary Text Encoder model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-filmgirl-ultra-text-encoder.fp16.safetensors"
