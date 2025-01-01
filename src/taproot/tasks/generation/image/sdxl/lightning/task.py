from __future__ import annotations

from typing import List, Optional, Union, TYPE_CHECKING

from taproot.constants import *

from ..base import StableDiffusionXLBase
from .scheduler import SDXLLightningScheduler
from .unet import (
    SDXLLightningUNet8Step,
    SDXLLightningUNet4Step,
    SDXLLightningUNet2Step
)

if TYPE_CHECKING:
    from torch import Tensor
    from taproot.hinting import SeedType, ImageType, ImageResultType

__all__ = [
    "StableDiffusionXLLightning8Step",
    "StableDiffusionXLLightning4Step",
    "StableDiffusionXLLightning2Step"
]

class StableDiffusionXLLightning8Step(StableDiffusionXLBase):
    """
    Stable Diffusion XL using the 8-step UNet.
    """

    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-lightning-8-step"
    default = False
    display_name = "Stable Diffusion XL Lightning (8-Step)"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "unet": SDXLLightningUNet8Step,
            "scheduler": SDXLLightningScheduler
        }
    }

    """Author Metadata"""
    author = "Shanchuan Lin"
    author_url = "https://arxiv.org/abs/2402.13929"
    author_additional = ["Anran Wang", "Xiao Yang"]
    author_affiliations = ["ByteDance Inc."]
    author_journal = "arXiv"
    author_journal_year = 2024
    author_journal_volume = "2402.13929"
    author_journal_title = "SDXL-Lightning: PRogressive Adversarial Diffusion Distillation"

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 0.0,
        guidance_rescale: float = 0.0,
        num_inference_steps: int = 8,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        image: Optional[ImageType] = None,
        mask_image: Optional[ImageType] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        denoising_end: Optional[float] = None,
        latents: Optional[Tensor] = None,
        prompt_embeds: Optional[Tensor] = None,
        negative_prompt_embeds: Optional[Tensor] = None,
        pooled_prompt_embeds: Optional[Tensor] = None,
        negative_pooled_prompt_embeds: Optional[Tensor] = None,
        #ip_adapter_image: Optional[ImageType] = None,
        #ip_adapter_image_embeds: Optional[List[Tensor]] = None,
        clip_skip: Optional[int] = None,
        seed: SeedType = None,
        scheduler: Optional[DIFFUSERS_SCHEDULER_LITERAL] = None,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL = "png",
        output_upload: bool = False
    ) -> ImageResultType:
        """
        Invokes SDXL lightning.

        Just pass through to the base class; we only duplicate for
        the purpose of updating defaults and documentation.
        """
        return super().__call__(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            image=image,
            mask_image=mask_image,
            timesteps=timesteps,
            sigmas=sigmas,
            denoising_end=denoising_end,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            #ip_adapter_image=ip_adapter_image,
            #ip_adapter_image_embeds=ip_adapter_image_embeds,
            clip_skip=clip_skip,
            seed=seed,
            scheduler=scheduler,
            output_format=output_format,
            output_upload=output_upload
        )

class StableDiffusionXLLightning4Step(StableDiffusionXLLightning8Step):
    """
    Stable Diffusion XL using the 4-step UNet.
    """
    model = "stable-diffusion-xl-lightning-4-step"
    display_name = "Stable Diffusion XL Lightning (4-Step)"
    pretrained_models = {
        **StableDiffusionXLLightning8Step.pretrained_models,
        **{
            "unet": SDXLLightningUNet4Step,
        }
    }

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 0.0,
        guidance_rescale: float = 0.0,
        num_inference_steps: int = 4,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        image: Optional[ImageType] = None,
        mask_image: Optional[ImageType] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        denoising_end: Optional[float] = None,
        latents: Optional[Tensor] = None,
        prompt_embeds: Optional[Tensor] = None,
        negative_prompt_embeds: Optional[Tensor] = None,
        pooled_prompt_embeds: Optional[Tensor] = None,
        negative_pooled_prompt_embeds: Optional[Tensor] = None,
        #ip_adapter_image: Optional[ImageType] = None,
        #ip_adapter_image_embeds: Optional[List[Tensor]] = None,
        clip_skip: Optional[int] = None,
        seed: SeedType = None,
        scheduler: Optional[DIFFUSERS_SCHEDULER_LITERAL] = None,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL = "png",
        output_upload: bool = False
    ) -> ImageResultType:
        """
        Invokes SDXL lightning.

        Just pass through to the base class; we only duplicate for
        the purpose of updating defaults and documentation.
        """
        return super().__call__(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            image=image,
            mask_image=mask_image,
            timesteps=timesteps,
            sigmas=sigmas,
            denoising_end=denoising_end,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            #ip_adapter_image=ip_adapter_image,
            #ip_adapter_image_embeds=ip_adapter_image_embeds,
            clip_skip=clip_skip,
            seed=seed,
            scheduler=scheduler,
            output_format=output_format,
            output_upload=output_upload
        )

class StableDiffusionXLLightning2Step(StableDiffusionXLLightning4Step):
    """
    Stable Diffusion XL using the 2-step UNet.
    """
    model = "stable-diffusion-xl-lightning-2-step"
    display_name = "Stable Diffusion XL Lightning (2-Step)"
    pretrained_models = {
        **StableDiffusionXLLightning4Step.pretrained_models,
        **{
            "unet": SDXLLightningUNet2Step,
        }
    }

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 0.0,
        guidance_rescale: float = 0.0,
        num_inference_steps: int = 2,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        image: Optional[ImageType] = None,
        mask_image: Optional[ImageType] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        denoising_end: Optional[float] = None,
        latents: Optional[Tensor] = None,
        prompt_embeds: Optional[Tensor] = None,
        negative_prompt_embeds: Optional[Tensor] = None,
        pooled_prompt_embeds: Optional[Tensor] = None,
        negative_pooled_prompt_embeds: Optional[Tensor] = None,
        #ip_adapter_image: Optional[ImageType] = None,
        #ip_adapter_image_embeds: Optional[List[Tensor]] = None,
        clip_skip: Optional[int] = None,
        seed: SeedType = None,
        scheduler: Optional[DIFFUSERS_SCHEDULER_LITERAL] = None,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL = "png",
        output_upload: bool = False
    ) -> ImageResultType:
        """
        Invokes SDXL lightning.

        Just pass through to the base class; we only duplicate for
        the purpose of updating defaults and documentation.
        """
        return super().__call__(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            image=image,
            mask_image=mask_image,
            timesteps=timesteps,
            sigmas=sigmas,
            denoising_end=denoising_end,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            #ip_adapter_image=ip_adapter_image,
            #ip_adapter_image_embeds=ip_adapter_image_embeds,
            clip_skip=clip_skip,
            seed=seed,
            scheduler=scheduler,
            output_format=output_format,
            output_upload=output_upload
        )
