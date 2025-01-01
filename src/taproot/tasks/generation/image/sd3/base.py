from __future__ import annotations

from typing import Dict, Optional, List, Union, Type, TYPE_CHECKING

from taproot.util import (
    is_multiple,
    log_duration,
)

from taproot.constants import *
from taproot.pretrained import (
    CLIPViTLTextEncoderWithProjection,
    CLIPViTLTokenizer,
    OpenCLIPViTGTextEncoder,
    OpenCLIPViTGTokenizer,
    T5XXLTextEncoder,
    T5XXLTextEncoderInt8,
    T5XXLTextEncoderNF4,
    T5XXLTokenizer,
)
from taproot.tasks.helpers import SpatialPromptInputType
from ..base import DiffusersTextToImageTask
from .pretrained import (
    StableDiffusion3VAE,
    StableDiffusion3Scheduler,
    StableDiffusion3Transformer,
    StableDiffusion35MediumTransformer,
    StableDiffusion35LargeTransformer,
    StableDiffusion35LargeTransformerInt8,
    StableDiffusion35LargeTransformerNF4,
)

if TYPE_CHECKING:
    import torch
    from diffusers.pipelines import DiffusionPipeline
    from taproot.hinting import ImageType, ImageResultType, SeedType

__all__ = [
    "StableDiffusion3",
    "StableDiffusion35Medium",
    "StableDiffusion35Large",
    "StableDiffusion35LargeInt8",
    "StableDiffusion35LargeNF4",
]

class StableDiffusion3(DiffusersTextToImageTask):
    """
    Stable Diffusion task using a v3 model.
    """

    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v3-medium"
    default = False
    display_name = "Stable Diffusion V3 (Medium) Image Generation"
    gpu_precision = "bfloat16"
    pretrained_models = {
        "vae": StableDiffusion3VAE,
        "transformer": StableDiffusion3Transformer,
        "scheduler": StableDiffusion3Scheduler,
        "text_encoder": CLIPViTLTextEncoderWithProjection,
        "text_encoder_2": OpenCLIPViTGTextEncoder,
        "text_encoder_3": T5XXLTextEncoder,
        "tokenizer": CLIPViTLTokenizer,
        "tokenizer_2": OpenCLIPViTGTokenizer,
        "tokenizer_3": T5XXLTokenizer,
    }

    """Authorship Metadata"""
    author = "Patrick Esser"
    author_url = "https://arxiv.org/abs/2403.03206"
    author_additional = ["Sumith Kulal", "Andreas Blattmann", "Rahim Entezari", "Jonas Müller", "Harry Saini", "Yam Levi", "Dominik Lorenz", "Axel Sauer", "Frederic Boesel", "Dustin Podell", "Tim Dockhorn", "Zion English", "Kyle Lacey", "Alex Goodwin", "Yannik Marek", "Robin Rombach"]
    author_affiliations = ["Stability AI"]
    author_journal = "arXiv"
    author_journal_title = "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    author_journal_volume = "2403.03206"
    author_journal_year = 2024

    """Licensing Metadata"""
    license = "Stability AI Community License Agreement"
    license_url = "https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE.md"
    license_attribution = False
    license_redistribution = True
    license_commercial = True
    license_derivatives = True
    license_hosting = True
    license_copy_left = False

    """Model Metadata"""
    static_memory_gb = 0.2 # 200 MB
    static_gpu_memory_gb = 17.86 # Measured on 3090

    """Metadata for Diffusers T2I"""
    use_compel = False
    model_type = "sd3"
    pag_applied_layers = ["blocks.(13|15|17)"]

    @classmethod
    def get_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the text-to-image pipeline.
        """
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
        return StableDiffusion3Pipeline # type: ignore[return-value]

    @classmethod
    def get_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the image-to-image pipeline.
        """
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import StableDiffusion3Img2ImgPipeline
        return StableDiffusion3Img2ImgPipeline # type: ignore[return-value]

    @classmethod
    def get_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the inpaint pipeline.
        """
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_inpaint import StableDiffusion3InpaintPipeline
        return StableDiffusion3InpaintPipeline # type: ignore[return-value]

    @classmethod
    def get_controlnet_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet text-to-image pipeline.
        """
        raise NotImplementedError(f"Controlnet text-to-image is not supported for {cls.__name__}.")

    @classmethod
    def get_controlnet_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet image-to-image pipeline.
        """
        raise NotImplementedError(f"Controlnet image-to-image is not supported for {cls.__name__}.")

    @classmethod
    def get_controlnet_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet inpaint pipeline.
        """
        raise NotImplementedError(f"Controlnet inpainting is not supported for {cls.__name__}.")

    @classmethod
    def get_pag_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the PAG text-to-image pipeline.
        """
        from diffusers.pipelines.pag.pipeline_pag_sd_3 import StableDiffusion3PAGPipeline
        return StableDiffusion3PAGPipeline # type: ignore[return-value]

    @classmethod
    def get_pag_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the PAG image-to-image pipeline.
        """
        from diffusers.pipelines.pag.pipeline_pag_sd_3_img2img import StableDiffusion3PAGImg2ImgPipeline # type: ignore[import-not-found,unused-ignore]
        return StableDiffusion3PAGImg2ImgPipeline # type: ignore[return-value,unused-ignore,no-any-return]

    @classmethod
    def get_pag_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the PAG inpaint pipeline.
        """
        raise NotImplementedError(f"PAG inpainting is not supported for {cls.__name__}.")

    @classmethod
    def get_pag_controlnet_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the PAG controlnet text-to-image pipeline.
        """
        raise NotImplementedError(f"PAG controlnet text-to-image is not supported for {cls.__name__}.")

    @classmethod
    def get_pag_controlnet_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the PAG controlnet image-to-image pipeline.
        """
        raise NotImplementedError(f"PAG controlnet image-to-image is not supported for {cls.__name__}.")

    @classmethod
    def get_pag_controlnet_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the PAG controlnet inpaint pipeline.
        """
        raise NotImplementedError(f"PAG controlnet inpainting is not supported for {cls.__name__}.")

    def get_pipeline_modules(self) -> Dict[str, torch.nn.Module]:
        """
        Get the pipeline modules.
        """
        return {
            "vae": self.pretrained.vae,
            "transformer": self.pretrained.transformer,
            "scheduler": self.pretrained.scheduler,
            "text_encoder": self.pretrained.text_encoder,
            "text_encoder_2": self.pretrained.text_encoder_2,
            "text_encoder_3": self.pretrained.text_encoder_3,
            "tokenizer": self.pretrained.tokenizer,
            "tokenizer_2": self.pretrained.tokenizer_2,
            "tokenizer_3": self.pretrained.tokenizer_3,
        }

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        image: Optional[ImageType] = None,
        mask_image: Optional[ImageType] = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 28,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
        seed: SeedType = None,
        max_sequence_length: int = 256,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL = "png",
        output_upload: bool = False,
        pag_scale: Optional[float] = None,
        pag_adaptive_scale: Optional[float] = None,
        highres_fix_factor: Optional[float] = 1.0,
        highres_fix_strength: Optional[float] = None,
        strength: Optional[float] = None,
        spatial_prompts: Optional[SpatialPromptInputType] = None,
        use_multidiffusion: bool = False,
        multidiffusion_tile_size: Optional[int] = 1024,
        multidiffusion_tile_stride: Optional[int] = 512,
        multidiffusion_mask_type: MULTIDIFFUSION_MASK_TYPE_LITERAL = DEFAULT_MULTIDIFFUSION_MASK_TYPE, # type: ignore[assignment]
    ) -> ImageResultType:
        """
        Invokes StableDiffusion 3.
        """
        with log_duration("inference"):
            results = self.invoke_pipeline(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_3=prompt_3,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                timesteps=timesteps,
                latents=latents,
                seed=seed,
                image=image,
                mask_image=mask_image,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                clip_skip=clip_skip,
                pag_scale=pag_scale,
                pag_adaptive_scale=pag_adaptive_scale,
                max_sequence_length=max_sequence_length,
                output_latent=output_format == "latent",
                highres_fix_factor=highres_fix_factor,
                highres_fix_strength=highres_fix_strength,
                strength=strength,
                spatial_prompts=spatial_prompts,
                use_multidiffusion=spatial_prompts is not None or use_multidiffusion,
                multidiffusion_tile_size=multidiffusion_tile_size,
                multidiffusion_tile_stride=multidiffusion_tile_stride,
                multidiffusion_mask_type=multidiffusion_mask_type,
            )

        return_first_item = num_images_per_prompt == 1 and not is_multiple(prompt)
        return self.get_output_from_image_result(
            results,
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=return_first_item
        )

class StableDiffusion35Medium(StableDiffusion3):
    """
    Stable Diffusion 3.5 by Stability AI
    """

    """Global Task Metadata"""
    model = "stable-diffusion-v3-5-medium"
    display_name = "Stable Diffusion V3.5 (Medium) Image Generation"
    static_gpu_memory_gb = 18.36 # Measured on 3090
    pretrained_models = {
        **StableDiffusion3.pretrained_models,
        **{
            "transformer": StableDiffusion35MediumTransformer,
        }
    }

class StableDiffusion35Large(StableDiffusion35Medium):
    """
    Stable Diffusion 3.5 by Stability AI
    """
    model = "stable-diffusion-v3-5-large"
    display_name = "Stable Diffusion V3.5 (Large) Image Generation"
    static_gpu_memory_gb = 31.36 # Guessed, don't have a big enough GPU to measure yet
    pretrained_models = {
        **StableDiffusion3.pretrained_models,
        **{
            "transformer": StableDiffusion35LargeTransformer,
        }
    }

class StableDiffusion35LargeInt8(StableDiffusion35Medium):
    """
    Stable Diffusion 3.5 by Stability AI with 8-bit quantization
    """
    model = "stable-diffusion-v3-5-large-int8"
    display_name = "Stable Diffusion V3.5 (Large) Image Generation (Int8)"
    static_gpu_memory_gb = 16.85 # Measured on 3090
    pretrained_models = {
        **StableDiffusion3.pretrained_models,
        **{
            "transformer": StableDiffusion35LargeTransformerInt8,
            "text_encoder_3": T5XXLTextEncoderInt8,
        }
    }

class StableDiffusion35LargeNF4(StableDiffusion35Medium):
    """
    Stable Diffusion 3.5 by Stability AI with 4-bit normalized floating point quantization (NF4)
    """
    model = "stable-diffusion-v3-5-large-nf4"
    display_name = "Stable Diffusion 3.5 (Large) Image Generation (NF4)"
    static_gpu_memory_gb = 12.99 # Measured on 3090
    pretrained_models = {
        **StableDiffusion3.pretrained_models,
        **{
            "transformer": StableDiffusion35LargeTransformerNF4,
            "text_encoder_3": T5XXLTextEncoderNF4,
        }
    }
