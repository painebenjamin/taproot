from __future__ import annotations

from typing import List, Optional, Union, TYPE_CHECKING

from taproot.constants import *

from ..base import StableDiffusionXLBase
from .scheduler import SDXLTurboScheduler
from .unet import SDXLTurboUNet
from .vae import SDXLTurboVAE

if TYPE_CHECKING:
    from torch import Tensor
    from taproot.hinting import SeedType, ImageType, ImageResultType

__all__ = ["StableDiffusionXLTurbo"]

class StableDiffusionXLTurbo(StableDiffusionXLBase):
    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl-turbo"
    default = False
    display_name = "Stable Diffusion XL Turbo Image Generation"
    pretrained_models = {
        **StableDiffusionXLBase.pretrained_models,
        **{
            "vae": SDXLTurboVAE,
            "unet": SDXLTurboUNet,
            "scheduler": SDXLTurboScheduler,
        }
    }

    """Authorship Metadata"""
    author = "Axel Sauer"
    author_additional = ["Dominik Lorenz", "Andreas Blattmann", "Robin Rombach"]
    author_affiliations = ["Stability AI"]
    author_url = "https://stability.ai/research/adversarial-diffusion-distillation"
    author_journal = "Stability AI Blog"
    author_journal_year = 2024
    author_journal_title = "Adversarial Diffusion Distillation"

    """Licensing Metadata"""
    license = "Stability AI Community License"
    license_url = "https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md"
    license_distribution = True
    license_attribution = True
    license_copy_left = False
    license_commercial = True
    license_derivatives = True
    license_hosting = True

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
        Invokes SDXL turbo.

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
