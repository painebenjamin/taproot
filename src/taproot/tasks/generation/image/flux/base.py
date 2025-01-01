from __future__ import annotations

from typing import Dict, Type, TYPE_CHECKING

from taproot.constants import *
from taproot.pretrained import (
    T5XXLTextEncoder,
    T5XXLTextEncoderInt8,
    T5XXLTextEncoderNF4,
    T5XXLTokenizer,
    CLIPViTLTextEncoder,
    CLIPViTLTokenizer,
)

from ..base import DiffusersTextToImageTask
from .pretrained import (
    FluxVAE,
    FluxScheduler,
)

if TYPE_CHECKING:
    import torch
    from diffusers.pipelines import DiffusionPipeline

__all__ = [
    "FluxBase",
    "FluxBaseInt8",
    "FluxBaseNF4"
]

class FluxBase(DiffusersTextToImageTask):
    """
    Image generation task using FLUX.1 Models
    """

    """Global Task Metadata"""
    pretrained_models = {
        "vae": FluxVAE,
        "scheduler": FluxScheduler,
        "text_encoder": CLIPViTLTextEncoder,
        "text_encoder_2": T5XXLTextEncoder,
        "tokenizer": CLIPViTLTokenizer,
        "tokenizer_2": T5XXLTokenizer,
    }
    gpu_precision = "bfloat16"
    static_memory_gb = .1806 # Measured on a 3090
    static_gpu_memory_gb = 29.5 # May be even more, idk, don't have a GPU to test on atm

    """Authorship Metadata"""
    author = "Black Forest Labs"
    author_url = "https://blackforestlabs.ai/announcing-black-forest-labs/"
    author_journal = "Black Forest Labs Blog"
    author_journal_year = 2024
    author_journal_title = "Announcing Black Forest Labs"

    """Licensing Metadata"""
    license = "FLUX.1 Non-Commercial License"
    license_url = "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md"
    license_attribution = True
    license_derivatives = True
    license_commercial = False
    license_redistribution = True
    license_copy_left = False
    license_hosting = True

    """Diffusers Metadata"""
    model_type = "flux"
    use_compel = False
    use_multidiffusion = False

    @classmethod
    def get_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the text-to-image pipeline.
        """
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
        return FluxPipeline # type: ignore[return-value]

    @classmethod
    def get_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the image-to-image pipeline.
        """
        from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline
        return FluxImg2ImgPipeline # type: ignore[return-value]

    @classmethod
    def get_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the inpaint pipeline.
        """
        from diffusers.pipelines.flux.pipeline_flux_inpaint import FluxInpaintPipeline
        return FluxInpaintPipeline # type: ignore[return-value]

    @classmethod
    def get_controlnet_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet text-to-image pipeline.
        """
        from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
        return FluxControlNetPipeline # type: ignore[return-value]

    @classmethod
    def get_controlnet_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet image-to-image pipeline.
        """
        from diffusers.pipelines.flux.pipeline_flux_controlnet_image_to_image import FluxControlNetImg2ImgPipeline
        return FluxControlNetImg2ImgPipeline # type: ignore[return-value]

    @classmethod
    def get_controlnet_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet inpaint pipeline.
        """
        from diffusers.pipelines.flux.pipeline_flux_controlnet_inpainting import FluxControlNetInpaintPipeline
        return FluxControlNetInpaintPipeline # type: ignore[return-value]

    def get_pipeline_modules(self) -> Dict[str, torch.nn.Module]:
        """
        Get the pipeline components.
        """
        if "transformer" not in self.pretrained_models:
            raise ValueError("Missing required pretrained model: transformer")

        return {
            "vae": self.pretrained.vae,
            "transformer": self.pretrained.transformer,
            "text_encoder": self.pretrained.text_encoder,
            "text_encoder_2": self.pretrained.text_encoder_2,
            "tokenizer": self.pretrained.tokenizer,
            "tokenizer_2": self.pretrained.tokenizer_2,
            "scheduler": self.pretrained.scheduler,
        }

class FluxBaseInt8(FluxBase):
    """
    Image generation task using FLUX.1 with Int-8 quantization on T5.
    """
    static_gpu_memory_gb = 21.22 # Measured on a 3090
    pretrained_models = {
        **FluxBase.pretrained_models,
        "text_encoder_2": T5XXLTextEncoderInt8,
    }

class FluxBaseNF4(FluxBase):
    """
    Image generation task using FLUX.1 with NF4 quantization on T5.
    """
    static_gpu_memory_gb = 14.36 # Measured on a 3090
    pretrained_models = {
        **FluxBase.pretrained_models,
        "text_encoder_2": T5XXLTextEncoderNF4,
    }
