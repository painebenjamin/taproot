from __future__ import annotations

from typing import Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from taproot.hinting import ImageType, SeedType, ImageResultType

from .base import (
    FluxBase,
    FluxBaseInt8,
    FluxBaseNF4
)
from .pretrained import (
    FluxSchnellTransformer,
    FluxSchnellTransformerInt8,
    FluxSchnellTransformerNF4
)

from taproot.constants import *
from taproot.util import (
    log_duration,
    is_multiple
)

__all__ = ["FluxSchnell"]

class FluxSchnell(FluxBase):
    """
    Image generation using FLUX.1 schnell.
    """

    """Global task metadata"""
    task: str = "image-generation"
    model: Optional[str] = "flux-v1-schnell"

    """Pretrained models"""
    pretrained_models = {
        **FluxBase.pretrained_models,
        **{"transformer": FluxSchnellTransformer}
    }

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: Optional[ImageType] = None,
        mask: Optional[ImageType] = None,
        guidance_scale: float = 0.0,
        num_inference_steps: int = 4,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        latents: Optional[Tensor] = None,
        prompt_embeds: Optional[Tensor] = None,
        pooled_prompt_embeds: Optional[Tensor] = None,
        seed: SeedType = None,
        max_sequence_length: int = 256,
        output_upload: bool = False,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL = "png",
        highres_fix_factor: Optional[float] = 1.0,
        highres_fix_strength: Optional[float] = None,
    ) -> ImageResultType:
        """
        Invokes FLUX.
        """
        with log_duration("inference"):
            results = self.invoke_pipeline(
                prompt=prompt,
                prompt_2=prompt_2,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                seed=seed,
                timesteps=timesteps,
                latents=latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                max_sequence_length=max_sequence_length,
                output_latent=output_format == "latent",
                highres_fix_factor=highres_fix_factor,
                highres_fix_strength=highres_fix_strength
            )

        return_first_item = num_images_per_prompt == 1 and not is_multiple(prompt)
        return self.get_output_from_image_result(
            results,
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=return_first_item
        )

class FluxSchnellInt8(FluxSchnell):
    """
    FLUX.1 schnell with 8-Bit integer quantization on the transformer and T5.
    """
    model: Optional[str] = "flux-v1-schnell-int8"
    static_gpu_memory_gb = FluxBaseInt8.static_gpu_memory_gb
    pretrained_models = {
        **FluxBaseInt8.pretrained_models,
        **{"transformer": FluxSchnellTransformerInt8}
    }

class FluxSchnellNF4(FluxSchnell):
    """
    FLUX.1 schnell with NF-4 quantization on the transformer and T5.
    """
    model: Optional[str] = "flux-v1-schnell-nf4"
    static_gpu_memory_gb = FluxBaseNF4.static_gpu_memory_gb
    pretrained_models = {
        **FluxBaseNF4.pretrained_models,
        **{"transformer": FluxSchnellTransformerNF4}
    }
