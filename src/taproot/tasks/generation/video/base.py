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
    logger,
    scale_tensor,
    to_bchw_tensor,
    get_aligned_timesteps_for_scheduler,
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

__all__ = ["DiffusersTextToVideoTask"]

class DiffusersTextToVideoTask(DiffusersPipelineTask):
    """
    A helper class for text-to-image tasks using Diffusers pipelines.

    These can be pretty varied, so a number of hooks are provided to allow for
    customization of the pipeline and the model handling.
    """
    default_steps = 50
    use_compel: bool = False # TODO: Add compel support for t2v
    default_negative_prompt: Optional[str] = "worst quality, inconsistent motion, blurry, jittery, distorted"

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
    def get_text_to_video_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the text-to-image pipeline.
        """
        raise NotImplementedError(f"Text-to-video is not supported for {cls.__name__}.")

    @classmethod
    def get_image_to_video_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the image-to-image pipeline.
        """
        raise NotImplementedError(f"Image-to-video is not supported for {cls.__name__}.")

    @classmethod
    def get_video_to_video_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the video-to-video pipeline.
        """
        raise NotImplementedError(f"Video-to-video is not supported for {cls.__name__}.")

    @classmethod
    def get_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the inpaint pipeline.
        """
        raise NotImplementedError(f"Inpainting is not supported for {cls.__name__}.")

    @classmethod
    def get_controlnet_text_to_video_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet text-to-video pipeline.
        """
        raise NotImplementedError(f"Controlnet text-to-video is not supported for {cls.__name__}.")

    @classmethod
    def get_controlnet_image_to_video_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet image-to-video pipeline.
        """
        raise NotImplementedError(f"Controlnet image-to-video is not supported for {cls.__name__}.")

    @classmethod
    def get_controlnet_video_to_video_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet video-to-video pipeline.
        """
        raise NotImplementedError(f"Controlnet video-to-video is not supported for {cls.__name__}.")

    @classmethod
    def get_controlnet_inpaint_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Get the controlnet inpaint pipeline.
        """
        raise NotImplementedError(f"Controlnet inpainting is not supported for {cls.__name__}.")

    """Shared Methods"""

    def get_pipeline_class(
        self,
        is_image_to_video: bool=False,
        is_video_to_video: bool=False,
        is_controlnet: bool=False,
        is_inpaint: bool=False,
        **kwargs: Any
    ) -> Type[DiffusionPipeline]:
        """
        Get the pipeline class.
        """
        if is_inpaint:
            if is_controlnet:
                return self.get_controlnet_inpaint_pipeline_class()
            return self.get_inpaint_pipeline_class()
        elif is_image_to_video:
            if is_controlnet:
                return self.get_controlnet_image_to_video_pipeline_class()
            return self.get_image_to_video_pipeline_class()
        elif is_video_to_video:
            if is_controlnet:
                return self.get_controlnet_video_to_video_pipeline_class()
            return self.get_video_to_video_pipeline_class()
        elif is_controlnet:
            return self.get_controlnet_text_to_video_pipeline_class()
        return self.get_text_to_video_pipeline_class()

    def invoke_pipeline(
        self,
        seed: Optional[SeedType]=None,
        timesteps: Optional[List[int]]=None,
        scheduler: Optional[DIFFUSERS_SCHEDULER_LITERAL]=None,
        image: Optional[ImageType]=None,
        mask_image: Optional[ImageType]=None,
        video: Optional[ImageType]=None,
        control_image: Optional[Dict[CONTROLNET_TYPE_LITERAL, ImageType]]=None,
        num_inference_steps: Optional[int]=None,
        output_latent: Optional[bool]=False,
        strength: Optional[float]=None,
        conditioning_strength: Optional[float]=None,
        height: Optional[int]=None,
        width: Optional[int]=None,
        use_multidiffusion: bool=False,
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
        if video is not None:
            video = to_bchw_tensor(video, num_channels=3).to(dtype=dtype, device=self.device)
            video = scale_tensor(video, round_to_nearest=8).clamp(0., 1.)
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
        is_image_to_video = image is not None and mask_image is None and (strength is None or strength < 1.0)
        is_video_to_video = video is not None and (strength is None or strength < 1.0)
        is_cfg = guidance_scale is not None and guidance_scale > 1.0

        pipeline = self.get_pipeline(
            scheduler=scheduler,
            is_image_to_video=is_image_to_video,
            is_video_to_video=is_video_to_video,
            is_controlnet=is_controlnet,
            is_inpaint=is_inpaint,
        )

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
        elif is_image_to_video and is_controlnet:
            kwargs["image"] = image
            kwargs["control_image"] = control_image
            kwargs["conditioning_strength"] = conditioning_strength or 1.0
            kwargs["strength"] = strength or 0.6
        elif is_image_to_video:
            kwargs["image"] = image
            kwargs["strength"] = strength or 0.6
        elif is_video_to_video:
            kwargs["video"] = video
            kwargs["strength"] = strength or 0.6
        elif is_controlnet:
            kwargs["image"] = control_image
            kwargs["conditioning_strength"] = conditioning_strength or 1.0

        if not is_image_to_video:
            kwargs["height"] = height
            kwargs["width"] = width
        elif is_image_to_video:
            kwargs["width"] = image.shape[-1] # type: ignore[union-attr]
            kwargs["height"] = image.shape[-2] # type: ignore[union-attr]
        elif is_video_to_video:
            kwargs["width"] = video.shape[-1]
            kwargs["height"] = video.shape[-2]

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
        accepts_frame_rate = "frame_rate" in invoke_signature.parameters

        if not accepts_frame_rate:
            kwargs.pop("frame_rate", None)

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

        if use_multidiffusion:
            spatial_prompts = self.get_spatial_prompts(spatial_prompts) if spatial_prompts is not None else None
            encoded_prompts = self.get_encoded_spatial_prompts(
                pipeline,
                kwargs=kwargs,
                clip_skip=clip_skip,
                accepts_negative_prompt=accepts_negative_prompt,
                spatial_prompts=spatial_prompts,
            )
            encoded_prompts.do_classifier_free_guidance = is_cfg
            encoded_prompts.do_perturbed_attention_guidance = False
            self.enable_multidiffusion(spatial_prompts=encoded_prompts)

        logger.debug(f"Invoke pipeline with kwargs: {kwargs}")

        try:
            result = pipeline( # type: ignore[operator]
                num_inference_steps=num_inference_steps,
                generator=self.get_generator(seed=seed),
                **kwargs
            ).frames
            return result # type: ignore[no-any-return]
        finally:
            self.disable_multidiffusion()
