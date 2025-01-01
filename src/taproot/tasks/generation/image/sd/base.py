from __future__ import annotations

from typing import Dict, Optional, List, Union, Type, TYPE_CHECKING

from taproot.util import (
    is_multiple,
    log_duration,
)
from taproot.constants import *
from taproot.pretrained import (
    CLIPViTLTextEncoder,
    CLIPViTLTokenizer,
)
from taproot.tasks.helpers import SpatialPromptInputType

from ..base import DiffusersTextToImageTask
from .pretrained import (
    StableDiffusionVAE,
    StableDiffusionUNet,
    StableDiffusionScheduler,
    StableDiffusionFeatureExtractor,
    StableDiffusionSafetyChecker
)

if TYPE_CHECKING:
    import torch
    from diffusers import DiffusionPipeline
    from taproot.hinting import ImageType, ImageResultType, SeedType

__all__ = ["StableDiffusionBase"]

DEFAULT_NUM_STEPS = 25
class StableDiffusionBase(DiffusersTextToImageTask):
    """
    Stable Diffusion task.
    """

    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-v1-5"
    default = False
    display_name = "Stable Diffusion v1.5 Image Generation"
    static_memory_gb = .191
    static_gpu_memory_gb = 2.58
    pretrained_models = {
        "vae": StableDiffusionVAE,
        "unet": StableDiffusionUNet,
        "tokenizer": CLIPViTLTokenizer,
        "text_encoder": CLIPViTLTextEncoder,
        "scheduler": StableDiffusionScheduler,
        "feature_extractor": StableDiffusionFeatureExtractor,
    }
    optional_pretrained_models = {
        "safety_checker": StableDiffusionSafetyChecker
    }

    """Authorship metadata"""
    author = "Robin Rombach"
    author_url = "https://arxiv.org/abs/2112.10752"
    author_additional = ["Andreas Blattmann", "Dominik Lorenz", "Patrick Esser", "Björn Ommer"]
    author_journal = "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
    author_journal_year = 2022
    author_journal_title = "High-Resolution Image Synthesis With Latent Diffusion Models"
    author_journal_pages = "10684-10695"

    """License Metadata"""
    license = LICENSE_OPENRAIL

    """Diffusers Metadata"""
    default_steps = DEFAULT_NUM_STEPS
    use_compel = True
    model_type = "sd"

    """Task-Specific Metadata"""
    use_safety_checker = False

    @classmethod
    def get_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the text-to-image pipeline.
        """
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
        return StableDiffusionPipeline # type: ignore[return-value]

    @classmethod
    def get_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the image-to-image pipeline.
        """
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
        return StableDiffusionImg2ImgPipeline # type: ignore[return-value]

    @classmethod
    def get_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the inpaint pipeline.
        """
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
        return StableDiffusionInpaintPipeline # type: ignore[return-value]

    @classmethod
    def get_controlnet_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet text-to-image pipeline.
        """
        from diffusers.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
        return StableDiffusionControlNetPipeline # type: ignore[return-value]

    @classmethod
    def get_controlnet_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet image-to-image pipeline.
        """
        from diffusers.pipelines.controlnet.pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
        return StableDiffusionControlNetImg2ImgPipeline # type: ignore[return-value]

    @classmethod
    def get_controlnet_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet inpaint pipeline.
        """
        from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
        return StableDiffusionControlNetInpaintPipeline # type: ignore[return-value]

    @classmethod
    def get_pag_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the PAG text-to-image pipeline.
        """
        from diffusers.pipelines.pag.pipeline_pag_sd import StableDiffusionPAGPipeline
        return StableDiffusionPAGPipeline # type: ignore[return-value]

    @classmethod
    def get_pag_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the PAG image-to-image pipeline.
        """
        from diffusers.pipelines.pag.pipeline_pag_sd_img2img import StableDiffusionPAGImg2ImgPipeline
        return StableDiffusionPAGImg2ImgPipeline # type: ignore[return-value]

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
        from diffusers.pipelines.pag.pipeline_pag_controlnet_sd import StableDiffusionControlNetPAGPipeline
        return StableDiffusionControlNetPAGPipeline # type: ignore[return-value]

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
        from diffusers.pipelines.pag.pipeline_pag_controlnet_sd_inpaint import StableDiffusionControlNetPAGInpaintPipeline
        return StableDiffusionControlNetPAGInpaintPipeline # type: ignore[return-value]

    def get_pipeline_modules(self) -> Dict[str, torch.nn.Module]:
        """
        Get the pipeline modules.
        """
        return {
            "vae": self.pretrained.vae,
            "unet": self.pretrained.unet,
            "text_encoder": self.pretrained.text_encoder,
            "tokenizer": self.pretrained.tokenizer,
            "scheduler": self.pretrained.scheduler,
            "feature_extractor": self.pretrained.feature_extractor,
            "safety_checker": None if not self.use_safety_checker else self.pretrained.safety_checker # type: ignore[dict-item]
        }

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        image: Optional[ImageType] = None,
        mask_image: Optional[ImageType] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        #ip_adapter_image: Optional[ImageType] = None,
        #ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        clip_skip: Optional[int] = None,
        seed: SeedType = None,
        strength: Optional[float] = None,
        scheduler: Optional[DIFFUSERS_SCHEDULER_LITERAL] = None,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL = "png",
        output_upload: bool = False,
        pag_scale: Optional[float] = None,
        pag_adaptive_scale: Optional[float] = None,
        highres_fix_factor: Optional[float] = 1.0,
        highres_fix_strength: Optional[float] = None,
        spatial_prompts: Optional[SpatialPromptInputType] = None,
        use_multidiffusion: bool = False,
        multidiffusion_tile_size: Optional[int] = 512,
        multidiffusion_tile_stride: Optional[int] = 256,
        multidiffusion_mask_type: MULTIDIFFUSION_MASK_TYPE_LITERAL = DEFAULT_MULTIDIFFUSION_MASK_TYPE, # type: ignore[assignment]
    ) -> ImageResultType:
        """
        Generate an image from text and/or image inputs using a Stable Diffusion V1.5 model.
        """
        with log_duration("inference"):
            results = self.invoke_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                timesteps=timesteps,
                sigmas=sigmas,
                latents=latents,
                seed=seed,
                image=image,
                mask_image=mask_image,
                strength=strength,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                #ip_adapter_image=ip_adapter_image,
                #ip_adapter_image_embeds=ip_adapter_image_embeds,
                clip_skip=clip_skip,
                pag_scale=pag_scale,
                pag_adaptive_scale=pag_adaptive_scale,
                output_latent=output_format == "latent",
                highres_fix_factor=highres_fix_factor,
                highres_fix_strength=highres_fix_strength,
                spatial_prompts=spatial_prompts,
                use_multidiffusion=spatial_prompts is not None or use_multidiffusion,
                multidiffusion_tile_size=multidiffusion_tile_size,
                multidiffusion_tile_stride=multidiffusion_tile_stride,
                multidiffusion_mask_type=multidiffusion_mask_type
            )

        return_first_item = num_images_per_prompt == 1 and not is_multiple(prompt)
        return self.get_output_from_image_result(
            results,
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=return_first_item
        )
