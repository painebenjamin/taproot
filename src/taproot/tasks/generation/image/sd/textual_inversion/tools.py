from .base import StableDiffusionPretrainedTextualInversion

__all__ = [
    "BadDreamStableDiffusionPretrainedTextualInversion",
    "UnrealisticDreamStableDiffusionPretrainedTextualInversion",
    "EasyNegativeStableDiffusionPretrainedTextualInversion"
]

class BadDreamStableDiffusionPretrainedTextualInversion(StableDiffusionPretrainedTextualInversion):
    name = "bad-dream"
    url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-textual-inversion-bad-dream.safetensors"
    author = "Lykon"
    author_url = "https://civitai.com/user/lykon"
    license = "OpenRAIL-M License with Addendum"
    license_attribution = False
    license_redistribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = True

class UnrealisticDreamStableDiffusionPretrainedTextualInversion(StableDiffusionPretrainedTextualInversion):
    name = "unrealistic-dream"
    url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-textual-inversion-unrealistic-dream.safetensors"
    author_url = "https://civitai.com/user/lykon"
    author = "Lykon"
    license = "OpenRAIL-M License with Addendum"
    license_attribution = False
    license_redistribution = True
    license_copy_left = False
    license_derivatives = False
    license_commercial = True
    license_hosting = True

class EasyNegativeStableDiffusionPretrainedTextualInversion(StableDiffusionPretrainedTextualInversion):
    name = "easynegative"
    url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-textual-inversion-easynegative.safetensors"
    author_url = "https://civitai.com/user/lykon"
    author = "rqdwdw"
    license = "OpenRAIL-M License with Addendum"
    license_attribution = False
    license_redistribution = True
    license_copy_left = True
    license_derivatives = True
    license_commercial = True
    license_hosting = True
