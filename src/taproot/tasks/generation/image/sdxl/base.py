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
    OpenCLIPViTGTextEncoder,
    OpenCLIPViTGTokenizer,
)
from taproot.tasks.helpers import SpatialPromptInputType

from ..base import DiffusersTextToImageTask
from .pretrained import (
    SDXLVAE,
    SDXLUNet,
    SDXLScheduler,
)

if TYPE_CHECKING:
    import torch
    from diffusers.pipelines import DiffusionPipeline
    from taproot.hinting import ImageType, ImageResultType, SeedType

__all__ = ["StableDiffusionXLBase"]

DEFAULT_NUM_STEPS = 20

class StableDiffusionXLBase(DiffusersTextToImageTask):
    """
    Stable Diffusion task using an XL model.
    """

    """Global Task Metadata"""
    task = "image-generation"
    model = "stable-diffusion-xl"
    default = False
    display_name = "Stable Diffusion XL Image Generation"
    pretrained_models = {
        "vae": SDXLVAE,
        "unet": SDXLUNet,
        "scheduler": SDXLScheduler,
        "text_encoder": CLIPViTLTextEncoder,
        "text_encoder_2": OpenCLIPViTGTextEncoder,
        "tokenizer": CLIPViTLTokenizer,
        "tokenizer_2": OpenCLIPViTGTokenizer,
    }
    static_memory_gb = .2158
    static_gpu_memory_gb = 7.06

    """Metadata for diffusers tasks"""
    default_steps = DEFAULT_NUM_STEPS
    use_compel = True
    model_type = "sdxl"

    """Authorship Metadata"""
    author = "Dustin Podell"
    author_url = "https://arxiv.org/abs/2307.01952"
    author_additional = ["Zion English", "Kyle Lacey", "Andreas Blattmann", "Tim Dockhorn", "Jonas Müller", "Joe Penna", "Robin Rombach"]
    author_affiliationas = ["Stability AI"]
    author_journal = "arXiv"
    author_journal_year = 2023
    author_journal_volume = "2307.01952"
    author_journal_title = "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis"

    """Licensing Metadata"""
    license = LICENSE_OPENRAILPP

    """Overrides for diffusers pipeline task"""

    @classmethod
    def get_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the text-to-image pipeline.
        """
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
        return StableDiffusionXLPipeline # type: ignore[return-value]

    @classmethod
    def get_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the image-to-image pipeline.
        """
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
        return StableDiffusionXLImg2ImgPipeline # type: ignore[return-value]

    @classmethod
    def get_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the inpaint pipeline.
        """
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import StableDiffusionXLInpaintPipeline
        return StableDiffusionXLInpaintPipeline # type: ignore[return-value]

    @classmethod
    def get_controlnet_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet text-to-image pipeline.
        """
        from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
        return StableDiffusionXLControlNetPipeline # type: ignore[return-value]

    @classmethod
    def get_controlnet_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet image-to-image pipeline.
        """
        from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl_img2img import StableDiffusionXLControlNetImg2ImgPipeline
        return StableDiffusionXLControlNetImg2ImgPipeline # type: ignore[return-value]

    @classmethod
    def get_controlnet_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet inpaint pipeline.
        """
        from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
        return StableDiffusionXLControlNetInpaintPipeline # type: ignore[return-value]

    @classmethod
    def get_pag_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the pag text-to-image pipeline.
        """
        from diffusers.pipelines.pag.pipeline_pag_sd_xl import StableDiffusionXLPAGPipeline
        return StableDiffusionXLPAGPipeline # type: ignore[return-value]

    @classmethod
    def get_pag_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the pag image-to-image pipeline.
        """
        from diffusers.pipelines.pag.pipeline_pag_sd_xl_img2img import StableDiffusionXLPAGImg2ImgPipeline
        return StableDiffusionXLPAGImg2ImgPipeline # type: ignore[return-value]

    @classmethod
    def get_pag_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the pag inpaint pipeline.
        """
        from diffusers.pipelines.pag.pipeline_pag_sd_xl_inpaint import StableDiffusionXLPAGInpaintPipeline
        return StableDiffusionXLPAGInpaintPipeline # type: ignore[return-value]

    @classmethod
    def get_pag_controlnet_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the pag controlnet text-to-image pipeline.
        """
        from diffusers.pipelines.pag.pipeline_pag_controlnet_sd_xl import StableDiffusionXLControlNetPAGPipeline
        return StableDiffusionXLControlNetPAGPipeline # type: ignore[return-value]

    @classmethod
    def get_pag_controlnet_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the pag controlnet image-to-image pipeline.
        """
        from diffusers.pipelines.pag.pipeline_pag_controlnet_sd_xl_img2img import StableDiffusionXLControlNetPAGImg2ImgPipeline
        return StableDiffusionXLControlNetPAGImg2ImgPipeline # type: ignore[return-value]

    @classmethod
    def get_pag_controlnet_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the pag controlnet inpaint pipeline.
        """
        raise NotImplementedError(f"PAG controlnet inpainting is not supported for {cls.__name__}.")

    def get_pipeline_modules(self) -> Dict[str, torch.nn.Module]:
        """
        Get the pipeline modules.
        """
        return {
            "vae": self.pretrained.vae,
            "unet": self.pretrained.unet,
            "text_encoder": self.pretrained.text_encoder,
            "text_encoder_2": self.pretrained.text_encoder_2,
            "tokenizer": self.pretrained.tokenizer,
            "tokenizer_2": self.pretrained.tokenizer_2,
            "scheduler": self.pretrained.scheduler
        }

    """Overrides for task"""

    def __call__( # type: ignore[override]
        self,
        *,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        image: Optional[ImageType] = None,
        mask_image: Optional[ImageType] = None,
        #control_image: Optional[Dict[CONTROLNET_TYPE_LITERAL, ImageType]] = None,
        guidance_scale: float = 5.0,
        guidance_rescale: float = 0.0,
        num_inference_steps: int = DEFAULT_NUM_STEPS,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        denoising_end: Optional[float] = None,
        strength: Optional[float] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        #ip_adapter_image: Optional[ImageType] = None,
        #ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        clip_skip: Optional[int] = None,
        seed: SeedType = None,
        pag_scale: Optional[float] = None,
        pag_adaptive_scale: Optional[float] = None,
        scheduler: Optional[DIFFUSERS_SCHEDULER_LITERAL] = None,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL = "png",
        output_upload: bool = False,
        highres_fix_factor: Optional[float] = 1.0,
        highres_fix_strength: Optional[float] = None,
        spatial_prompts: Optional[SpatialPromptInputType] = None,
        use_multidiffusion: bool = False,
        multidiffusion_tile_size: Optional[int] = 1024,
        multidiffusion_tile_stride: Optional[int] = 512,
        multidiffusion_mask_type: MULTIDIFFUSION_MASK_TYPE_LITERAL = DEFAULT_MULTIDIFFUSION_MASK_TYPE, # type: ignore[assignment]
    ) -> ImageResultType:
        """
        Generate an image from text and/or images using a stable diffusion XL model.
        """
        with log_duration("inference"):
            results = self.invoke_pipeline(
                image=image,
                mask_image=mask_image,
                #control_image=control_image,
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
                timesteps=timesteps,
                sigmas=sigmas,
                denoising_end=denoising_end,
                latents=latents,
                strength=strength,
                seed=seed,
                pag_scale=pag_scale,
                pag_adaptive_scale=pag_adaptive_scale,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                #ip_adapter_image=ip_adapter_image,
                #ip_adapter_image_embeds=ip_adapter_image_embeds,
                clip_skip=clip_skip,
                output_latent=output_format == "latent",
                highres_fix_factor=highres_fix_factor,
                highres_fix_strength=highres_fix_strength,
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
