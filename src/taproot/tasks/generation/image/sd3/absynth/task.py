from ..base import (
    StableDiffusion35Large,
    StableDiffusion35LargeInt8,
    StableDiffusion35LargeNF4,
)

from .transformer import (
    StableDiffusion35LargeAbsynthV19Transformer,
    StableDiffusion35LargeAbsynthV19TransformerInt8,
    StableDiffusion35LargeAbsynthV19TransformerNF4,
)

__all__ = [
    "StableDiffusion35LargeAbsynthV19",
    "StableDiffusion35LargeAbsynthV19Int8",
    "StableDiffusion35LargeAbsynthV19NF4",
]

class StableDiffusion35LargeAbsynthV19(StableDiffusion35Large):
    model = "stable-diffusion-v3-5-large-absynth-v1-9"
    pretrained_models = {
        **StableDiffusion35Large.pretrained_models,
        **{"transformer": StableDiffusion35LargeAbsynthV19Transformer},
    }

class StableDiffusion35LargeAbsynthV19Int8(StableDiffusion35LargeInt8):
    model = "stable-diffusion-v3-5-large-absynth-v1-9-int8"
    pretrained_models = {
        **StableDiffusion35LargeInt8.pretrained_models,
        **{"transformer": StableDiffusion35LargeAbsynthV19TransformerInt8},
    }

class StableDiffusion35LargeAbsynthV19NF4(StableDiffusion35LargeNF4):
    model = "stable-diffusion-v3-5-large-absynth-v1-9-nf4"
    pretrained_models = {
        **StableDiffusion35LargeNF4.pretrained_models,
        **{"transformer": StableDiffusion35LargeAbsynthV19TransformerNF4},
    }
