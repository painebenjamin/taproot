from __future__ import annotations

import inspect
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TYPE_CHECKING
)
from taproot.util import (
    get_aligned_timesteps_for_scheduler,
    get_seed,
    logger,
    scale_tensor,
    to_bchw_tensor,
)
from taproot.constants import *
from taproot.tasks.helpers import (
    DiffusersPipelineTask,
    SpatialPromptInputType,
)

if TYPE_CHECKING:
    import torch
    from diffusers.pipelines import DiffusionPipeline
    from taproot.hinting import ImageType, SeedType

__all__ = ["DiffusersTextToImageTask"]

class DiffusersTextToImageTask(DiffusersPipelineTask):
    """
    A helper class for text-to-image tasks using Diffusers pipelines.

    These can be pretty varied, so a number of hooks are provided to allow for
    customization of the pipeline and the model handling.
    """
    use_compel: bool = True
    use_multidiffusion: bool = True
    default_steps: int = 25

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Required packages.
        """
        return {
            "pil": PILLOW_VERSION_SPEC,
            "torch": TORCH_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
            "diffusers": DIFFUSERS_VERSION_SPEC,
            "torchvision": TORCHVISION_VERSION_SPEC,
            "transformers": TRANSFORMERS_VERSION_SPEC,
            "safetensors": SAFETENSORS_VERSION_SPEC,
            "accelerate": ACCELERATE_VERSION_SPEC,
            "sklearn": SKLEARN_VERSION_SPEC,
            "sentencepiece": SENTENCEPIECE_VERSION_SPEC,
            "compel": COMPEL_VERSION_SPEC,
            "peft": PEFT_VERSION_SPEC,
        }

    """Classmethod stubs"""

    @classmethod
    def get_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the text-to-image pipeline.
        """
        raise NotImplementedError(f"Text-to-image is not supported for {cls.__name__}.")

    @classmethod
    def get_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the image-to-image pipeline.
        """
        raise NotImplementedError(f"Image-to-image is not supported for {cls.__name__}.")

    @classmethod
    def get_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the inpaint pipeline.
        """
        raise NotImplementedError(f"Inpainting is not supported for {cls.__name__}.")

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
        Get the pag text-to-image pipeline.
        """
        raise NotImplementedError(f"PAG text-to-image is not supported for {cls.__name__}.")

    @classmethod
    def get_pag_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the pag image-to-image pipeline.
        """
        raise NotImplementedError(f"PAG image-to-image is not supported for {cls.__name__}.")

    @classmethod
    def get_pag_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the pag inpaint pipeline.
        """
        raise NotImplementedError(f"PAG inpainting is not supported for {cls.__name__}.")

    @classmethod
    def get_pag_controlnet_text_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the pag controlnet text-to-image pipeline.
        """
        raise NotImplementedError(f"PAG controlnet text-to-image is not supported for {cls.__name__}.")

    @classmethod
    def get_pag_controlnet_image_to_image_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the pag controlnet image-to-image pipeline.
        """
        raise NotImplementedError(f"PAG controlnet image-to-image is not supported for {cls.__name__}.")

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
        raise NotImplementedError(f"Pipeline modules not configured for {type(self).__name__}.")

    """Shared Methods"""

    def get_pipeline_kwargs(
        self,
        is_image_to_image: bool=False,
        is_controlnet: bool=False,
        is_inpaint: bool=False,
        is_pag: bool=False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Get the pipeline kwargs.
        """
        kwargs = super().get_pipeline_kwargs(**kwargs)
        if is_pag and self.pag_applied_layers is not None:
            kwargs["pag_applied_layers"] = self.pag_applied_layers
        return kwargs

    def get_pipeline_class(
        self,
        is_image_to_image: bool=False,
        is_controlnet: bool=False,
        is_inpaint: bool=False,
        is_pag: bool=False,
        **kwargs: Any
    ) -> Type[DiffusionPipeline]:
        """
        Get the pipeline class.
        """
        if is_inpaint:
            if is_controlnet:
                if is_pag:
                    return self.get_pag_controlnet_inpaint_pipeline_class()
                else:
                    return self.get_controlnet_inpaint_pipeline_class()
            elif is_pag:
                return self.get_pag_inpaint_pipeline_class()
            else:
                return self.get_inpaint_pipeline_class()
        elif is_image_to_image:
            if is_controlnet:
                if is_pag:
                    return self.get_pag_controlnet_image_to_image_pipeline_class()
                else:
                    return self.get_controlnet_image_to_image_pipeline_class()
            elif is_pag:
                return self.get_pag_image_to_image_pipeline_class()
            else:
                return self.get_image_to_image_pipeline_class()
        elif is_controlnet:
            if is_pag:
                return self.get_pag_controlnet_text_to_image_pipeline_class()
            else:
                return self.get_controlnet_text_to_image_pipeline_class()
        elif is_pag:
            return self.get_pag_text_to_image_pipeline_class()
        return self.get_text_to_image_pipeline_class()

    def invoke_pipeline(
        self,
        seed: Optional[SeedType]=None,
        timesteps: Optional[List[int]]=None,
        scheduler: Optional[DIFFUSERS_SCHEDULER_LITERAL]=None,
        image: Optional[ImageType]=None,
        mask_image: Optional[ImageType]=None,
        control_image: Optional[Dict[CONTROLNET_TYPE_LITERAL, ImageType]]=None,
        num_inference_steps: Optional[int]=None,
        output_latent: Optional[bool]=False,
        strength: Optional[float]=None,
        conditioning_strength: Optional[float]=None,
        pag_scale: Optional[float]=None,
        pag_adaptive_scale: Optional[float]=None,
        height: Optional[int]=None,
        width: Optional[int]=None,
        use_multidiffusion: bool=True,
        multidiffusion_tile_size: Optional[int]=None,
        multidiffusion_tile_stride: Optional[int]=None,
        multidiffusion_mask_type: MULTIDIFFUSION_MASK_TYPE_LITERAL=DEFAULT_MULTIDIFFUSION_MASK_TYPE, # type: ignore[assignment]
        highres_fix_factor: Optional[float]=1.0,
        highres_fix_strength: Optional[float]=None,
        clip_skip: Optional[int]=None,
        spatial_prompts: Optional[SpatialPromptInputType]=None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Invoke the pipeline.
        """
        import torch

        autoencoding_model = self.get_autoencoding_model()
        if autoencoding_model is not None:
            dtype = next(autoencoding_model.parameters()).dtype
        else:
            dtype = self.dtype

        if image is not None:
            image = to_bchw_tensor(image, num_channels=3).to(dtype=dtype, device=self.device)
            image = scale_tensor(image, round_to_nearest=8).clamp(0., 1.)
        if mask_image is not None:
            mask_image = to_bchw_tensor(mask_image, num_channels=1).to(dtype=dtype, device=self.device)
            mask_image = scale_tensor(mask_image, round_to_nearest=8).clamp(0., 1.)
        if control_image is not None:
            control_image = {
                key: scale_tensor(
                    to_bchw_tensor(value, num_channels=3).to(dtype=dtype, device=self.device),
                    round_to_nearest=8
                ).clamp(0., 1.)
                for key, value in control_image.items()
            }

        guidance_scale = kwargs.get("guidance_scale", None)

        is_inpaint = image is not None and mask_image is not None
        is_controlnet = control_image is not None
        is_image_to_image = image is not None and mask_image is None and (strength is None or strength < 1.0)
        is_pag = pag_scale is not None and pag_scale > 0.0
        is_cfg = guidance_scale is not None and guidance_scale > 1.0

        use_high_res_fix = (
            highres_fix_factor is not None and highres_fix_factor > 0.0 and
            highres_fix_strength is not None and highres_fix_strength > 0.0
        )

        pipeline = self.get_pipeline(
            scheduler=scheduler,
            is_image_to_image=is_image_to_image,
            is_controlnet=is_controlnet,
            is_inpaint=is_inpaint,
            is_pag=is_pag,
        )

        generator = torch.Generator(device=self.device)
        generator.manual_seed(get_seed(seed))

        if is_inpaint and is_controlnet:
            kwargs["image"] = image
            kwargs["mask_image"] = mask_image
            kwargs["control_image"] = control_image
            kwargs["strength"] = strength or 1.0
            kwargs["conditioning_strength"] = conditioning_strength or 1.0
        elif is_inpaint:
            kwargs["image"] = image
            kwargs["mask_image"] = mask_image
            kwargs["strength"] = strength or 1.0
        elif is_image_to_image and is_controlnet:
            kwargs["image"] = image
            kwargs["control_image"] = control_image
            kwargs["conditioning_strength"] = conditioning_strength or 1.0
            kwargs["strength"] = strength or 0.6
        elif is_image_to_image:
            kwargs["image"] = image
            kwargs["strength"] = strength or 0.6
        elif is_controlnet:
            kwargs["image"] = control_image
            kwargs["conditioning_strength"] = conditioning_strength or 1.0

        if is_pag:
            kwargs["pag_scale"] = pag_scale
            kwargs["pag_adaptive_scale"] = pag_adaptive_scale or 0.0

        if not is_image_to_image:
            kwargs["height"] = height
            kwargs["width"] = width
        else:
            kwargs["width"] = image.shape[-1] # type: ignore[union-attr]
            kwargs["height"] = image.shape[-2] # type: ignore[union-attr]

        if num_inference_steps is None:
            num_inference_steps = self.default_steps
        if timesteps is None and self.model_type is not None:
            timesteps = get_aligned_timesteps_for_scheduler(
                pipeline.scheduler, # type: ignore[attr-defined]
                model_type=self.model_type, # type: ignore[arg-type]
                num_timesteps=num_inference_steps,
            )
        if timesteps is not None:
            kwargs["timesteps"] = timesteps

        invoke_signature = inspect.signature(pipeline.__call__) # type: ignore[operator]
        accepts_clip_skip = "clip_skip" in invoke_signature.parameters
        accepts_output_type = "output_type" in invoke_signature.parameters
        accepts_output_format = "output_format" in invoke_signature.parameters
        accepts_negative_prompt = "negative_prompt" in invoke_signature.parameters

        if accepts_output_type:
            kwargs["output_type"] = "latent" if output_latent else "pt"
        elif accepts_output_format:
            kwargs["output_format"] = "latent" if output_latent else "pt"
        else:
            raise ValueError("Pipeline does not accept output type or format - is this a legacy Diffusers pipeline?")

        ignored_kwargs = set(kwargs.keys()) - set(invoke_signature.parameters.keys())
        if ignored_kwargs:
            logger.warning(f"Ignoring unknown kwargs: {ignored_kwargs}")
            for key in ignored_kwargs:
                del kwargs[key]

        if accepts_negative_prompt and kwargs.get("negative_prompt", None) is None:
            kwargs["negative_prompt"] = self.default_negative_prompt

        if self.use_compel:
            self.compile_prompts_into_kwargs(
                pipeline,
                kwargs,
                clip_skip=clip_skip,
                accepts_negative_prompt=accepts_negative_prompt,
            )
        elif accepts_clip_skip:
            kwargs["clip_skip"] = clip_skip

        if use_multidiffusion and self.use_multidiffusion:
            spatial_prompts = self.get_spatial_prompts(spatial_prompts) if spatial_prompts is not None else None
            encoded_prompts = self.get_encoded_spatial_prompts(
                pipeline,
                kwargs=kwargs,
                clip_skip=clip_skip,
                accepts_negative_prompt=accepts_negative_prompt,
                spatial_prompts=spatial_prompts,
            )
            encoded_prompts.do_classifier_free_guidance = is_cfg
            encoded_prompts.do_perturbed_attention_guidance = is_pag
            self.enable_multidiffusion(
                spatial_prompts=encoded_prompts,
                mask_type=multidiffusion_mask_type,
                tile_size=None if multidiffusion_tile_size is None else int(multidiffusion_tile_size // 8),
                tile_stride=None if multidiffusion_tile_stride is None else int(multidiffusion_tile_stride // 8),
            )

        logger.debug(f"Invoke pipeline with kwargs: {kwargs}")

        try:
            result = pipeline( # type: ignore[operator]
                num_inference_steps=num_inference_steps,
                generator=generator,
                **kwargs
            ).images

            if use_high_res_fix:
                for ignored_kwarg in ["image", "mask_image", "control_image", "latents", "width", "height", "strength"]:
                    kwargs.pop(ignored_kwarg, None)
                if accepts_output_type and not output_latent:
                    kwargs["output_type"] = "pt"
                elif accepts_output_format and not output_latent:
                    kwargs["output_format"] = "pt"

                pipeline = self.get_pipeline(
                    scheduler=scheduler,
                    is_image_to_image=True,
                    is_controlnet=False,
                    is_inpaint=False,
                    is_pag=is_pag,
                )

                i2i_signature = inspect.signature(pipeline.__call__) # type: ignore[operator]
                accepts_width = "width" in i2i_signature.parameters
                accepts_height = "height" in i2i_signature.parameters

                result = scale_tensor(
                    result,
                    scale_factor=1.0 + (highres_fix_factor or 0.0)
                ).clamp(0., 1.)

                if accepts_width and accepts_height:
                    b, c, h, w = result.shape
                    kwargs["width"] = w
                    kwargs["height"] = h

                result = pipeline( # type: ignore[operator]
                    image=result,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    strength=highres_fix_strength,
                    **kwargs
                ).images

                result = scale_tensor(
                    result,
                    scale_factor=1.0 / (1.0 + (highres_fix_factor or 0.0))
                ).clamp(0., 1.)

            return result # type: ignore[no-any-return]
        finally:
            self.disable_multidiffusion()
