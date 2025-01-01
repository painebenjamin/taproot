from __future__ import annotations

import os
import re
import json

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    TYPE_CHECKING
)
from typing_extensions import Literal
from taproot.util import (
    disable_2d_multidiffusion,
    enable_2d_multidiffusion,
    encode_prompt_for_model,
    get_diffusers_scheduler_by_name,
    get_seed,
    logger,
    maybe_use_tqdm,
    wrap_module_forward_dtype,
    unwrap_module_forward_dtype,
    SpatioTemporalPrompt,
    EncodedPrompts,
    EncodedPrompt,
)
from taproot.constants import *
from taproot.tasks.base import Task

if TYPE_CHECKING:
    import torch
    from diffusers.schedulers.scheduling_utils import SchedulerMixin
    from diffusers.pipelines import DiffusionPipeline
    from taproot.hinting import SeedType

__all__ = [
    "DiffusersPipelineTask",
    "SpatialPromptType",
    "SpatialPromptInputType",
]

SpatialPromptType = Union[str, Dict[str, Any], SpatioTemporalPrompt]
SpatialPromptInputType = Union[SpatialPromptType, Sequence[SpatialPromptType]]

class DiffusersPipelineTask(Task):
    """
    A helper class for media generation tasks using Diffusers pipelines.

    These can be pretty varied, so a number of hooks are provided to allow for
    customization of the pipeline and the model handling.
    """
    use_compel: bool = True
    wrap_dtype_mismatch: bool = False
    model_type: Optional[str] = None
    autoencoding_model_name: Optional[str] = None
    denoising_model_name: Optional[str] = None
    pag_applied_layers: Optional[List[str]] = None
    default_negative_prompt: Optional[str] = "lowres, blurry, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

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

    def get_offload_models(self) -> Union[List[str], bool]:
        """
        Get offload models.
        """
        if self.pretrained_models is not None and (
            self.enable_model_offload or
            self.enable_sequential_offload
        ):
            return [
                name for name in self.pretrained_models.keys()
                if "tokenizer" not in name
                and "scheduler" not in name
                and getattr(self.pretrained_models[name], "quantization", None) is None
            ]
        return False

    """Method Stubs"""

    def get_pipeline_class(self, **kwargs: Any) -> Type[DiffusionPipeline]:
        """
        Get the pipeline class.
        """
        raise NotImplementedError(f"Pipeline class not configured for {type(self).__name__}.")

    def get_pipeline_modules(self) -> Dict[str, torch.nn.Module]:
        """
        Get the pipeline modules.
        """
        raise NotImplementedError(f"Pipeline modules not configured for {type(self).__name__}.")

    def get_pipeline_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Get the pipeline kwargs.
        """
        return {}

    """Shared Methods"""

    def get_pipeline(self, **kwargs: Any) -> DiffusionPipeline:
        """
        Get the pipeline.
        """
        pipeline_class = self.get_pipeline_class(**kwargs)
        pipeline_modules = self.get_pipeline_modules()
        pipeline_kwargs = self.get_pipeline_kwargs(**kwargs)

        pipeline_modules["scheduler"] = self.get_scheduler( # type: ignore[assignment]
            scheduler_name=kwargs.get("scheduler", None),
            scheduler=pipeline_modules.get("scheduler", None), # type: ignore[arg-type]
        )

        pipeline = pipeline_class(**{**pipeline_modules, **pipeline_kwargs})
        vae = self.get_autoencoding_model()
        denoising_model = self.get_denoising_model()

        if vae is not None and denoising_model is not None and self.wrap_dtype_mismatch:
            vae_dtype = next(vae.parameters()).dtype
            denoising_dtype = next(denoising_model.parameters()).dtype
            if vae_dtype != denoising_dtype:
                logger.debug(f"Wrapping denoising model forward to match VAE dtype for {type(self).__name__}.")
                wrap_module_forward_dtype(denoising_model, input_dtype=denoising_dtype, output_dtype=vae_dtype)
            else:
                unwrap_module_forward_dtype(denoising_model)

        if self.enable_encode_tiling:
            if vae is None:
                logger.warning(f"No VAE found for {type(self).__name__}, cannot enable tiling.")
            else:
                logger.debug(f"Enabling VAE tiling for {type(self).__name__}.")
                vae.enable_tiling()
        if self.enable_encode_slicing:
            if vae is None:
                logger.warning(f"No VAE found for {type(self).__name__}, cannot enable slicing.")
            else:
                logger.debug(f"Enabling VAE slicing for {type(self).__name__}.")
                vae.enable_slicing()
        if self.enable_model_offload:
            if hasattr(pipeline, "enable_model_cpu_offload"):
                logger.debug(f"Enabling model CPU offload for {type(self).__name__}.")
                pipeline.enable_model_cpu_offload()
            else:
                logger.warning(f"Model CPU offload not supported for {type(self).__name__}.")
        elif self.enable_sequential_offload:
            if hasattr(pipeline, "enable_sequential_cpu_offload"):
                logger.debug(f"Enabling sequential CPU offload for {type(self).__name__}.")
                pipeline.enable_sequential_cpu_offload()
            else:
                logger.warning(f"Sequential CPU offload not supported for {type(self).__name__}.")

        return pipeline

    def get_autoencoding_model(self) -> torch.nn.Module:
        """
        Get the autoencoding model.
        """
        if self.autoencoding_model_name is not None:
            return getattr(self.pretrained, self.autoencoding_model_name) # type: ignore[no-any-return]
        vae = getattr(self.pretrained, "vae", None)
        if vae is not None:
            return vae # type: ignore[no-any-return]
        vqgan = getattr(self.pretrained, "vqgan", None)
        if vqgan is not None:
            return vqgan # type: ignore[no-any-return]
        raise ValueError(f"No autoencoding model name set, and could not find VAE or VQGAN for {type(self).__name__}")

    def get_denoising_model(self) -> torch.nn.Module:
        """
        Get the denoising model.
        """
        if self.denoising_model_name is not None:
            return getattr(self.pretrained, self.denoising_model_name) # type: ignore[no-any-return]
        transformer = getattr(self.pretrained, "transformer", None)
        if transformer is not None:
            return transformer # type: ignore[no-any-return]
        unet = getattr(self.pretrained, "unet", None)
        if unet is not None:
            return unet # type: ignore[no-any-return]
        raise ValueError(f"No denoising model name set, and could not find transformer or unet for {type(self).__name__}")

    def enable_multidiffusion(
        self,
        spatial_prompts: Optional[EncodedPrompts]=None,
        tile_size: Optional[Union[int, Tuple[int, int]]]=None,
        tile_stride: Optional[Union[int, Tuple[int, int]]]=None,
        use_tqdm: bool=False,
        mask_type: Literal["constant", "bilinear", "gaussian"]="bilinear",
    ) -> None:
        """
        Enable multidiffusion.
        """
        enable_2d_multidiffusion(
            self.get_denoising_model(),
            spatial_prompts=spatial_prompts,
            tile_size=tile_size,
            tile_stride=tile_stride,
            use_tqdm=use_tqdm,
            mask_type=mask_type,
        )

    def disable_multidiffusion(self) -> None:
        """
        Disable multidiffusion.
        """
        disable_2d_multidiffusion(self.get_denoising_model())

    def get_scheduler(
        self,
        scheduler_name: Optional[DIFFUSERS_SCHEDULER_LITERAL]=None,
        scheduler: Optional[SchedulerMixin]=None,
    ) -> SchedulerMixin:
        """
        Gets the scheduler.
        """
        if scheduler_name is not None:
            return get_diffusers_scheduler_by_name(scheduler_name, scheduler.config if scheduler is not None else None) # type: ignore[attr-defined]
        elif scheduler is not None:
            return scheduler
        raise ValueError("No scheduler provided, and no default available. Add a pretrained scheduler to your task configuration.")

    def get_prompts_from_kwargs(
        self,
        key_text: str="prompt",
        **kwargs: Any
    ) -> List[str]:
        """
        Get prompts from kwargs.
        """
        prompts = {}
        for key, value in kwargs.items():
            if key.startswith(key_text) and "embeds" not in key and value is not None:
                key_parts = key[len(key_text)+1:].split("_")
                if len(key_parts) == 1:
                    prompt_index = int(key_parts[0]) - 1 if key_parts[0] else 0
                    prompts[prompt_index] = value

        return [prompts[i] for i in sorted(prompts.keys())]

    def get_negative_prompts_from_kwargs(
        self,
        key_text: str="negative_prompt",
        **kwargs: Any
    ) -> List[str]:
        """
        Get negative prompts from kwargs.
        """
        return self.get_prompts_from_kwargs(key_text=key_text, **kwargs)

    def get_compiled_prompt_embeds(
        self,
        pipeline: DiffusionPipeline,
        prompts: List[str],
        negative_prompts: Optional[List[str]]=None,
        clip_skip: Optional[int]=None,
    ) -> Optional[
        Tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ]
    ]:
        """
        Compiles prompts using compel.
        """
        import torch

        num_prompts = len(prompts)
        num_negative_prompts = 0 if negative_prompts is None else len(negative_prompts)
        num_text_encoders = 0

        if getattr(pipeline, "text_encoder_3", None) is not None:
            num_text_encoders = 3
        elif getattr(pipeline, "text_encoder_2", None) is not None:
            num_text_encoders = 2
        elif getattr(pipeline, "text_encoder", None) is not None:
            num_text_encoders = 1

        if num_text_encoders == 0:
            logger.warning("No text encoders found in pipeline - compel will not be applied.")
            return None
        elif num_prompts == 0:
            logger.warning("No prompts found - compel will not be applied.")
            return None

        text_encoders = [pipeline.text_encoder] # type: ignore[attr-defined]
        tokenizers = [pipeline.tokenizer] # type: ignore[attr-defined]
        for i in range(num_text_encoders - 1):
            text_encoders.append(getattr(pipeline, f"text_encoder_{i+2}"))
            tokenizers.append(getattr(pipeline, f"tokenizer_{i+2}"))

        # encode prompts
        encoded_prompt_embeds = []
        encoded_negative_prompt_embeds = []

        encoded_pooled_prompt_embeds = []
        encoded_negative_pooled_prompt_embeds = []

        for i in range(num_text_encoders):
            is_offloaded = False
            if next(text_encoders[i].parameters()).device.type == "cpu":
                is_offloaded = True
                logger.debug(f"Moving offloaded text encoder {i+1} to {self.device} with dtype {self.dtype} for compel.")
                text_encoders[i].to(self.device, dtype=self.dtype)

            prompt = prompts[i] if num_prompts > i else prompts[-1]
            encoded = encode_prompt_for_model(
                model_type=self.model_type, # type: ignore[arg-type]
                prompt=prompt,
                tokenizer=tokenizers[i],
                text_encoder=text_encoders[i],
                clip_skip=clip_skip,
                device="cpu"
            )

            if isinstance(encoded, tuple):
                prompt_embeds, pooled_prompt_embeds = encoded
            else:
                prompt_embeds = encoded
                pooled_prompt_embeds = None

            encoded_prompt_embeds.append(prompt_embeds)
            if pooled_prompt_embeds is not None:
                encoded_pooled_prompt_embeds.append(pooled_prompt_embeds)

            if num_negative_prompts > 0:
                negative_prompt = negative_prompts[i] if num_negative_prompts > i else negative_prompts[-1] # type: ignore[index]
                encoded = encode_prompt_for_model(
                    model_type=self.model_type, # type: ignore[arg-type]
                    prompt=negative_prompt,
                    tokenizer=tokenizers[i],
                    text_encoder=text_encoders[i],
                    clip_skip=clip_skip,
                    device="cpu"
                )
                if isinstance(encoded, tuple):
                    negative_prompt_embeds, negative_pooled_prompt_embeds = encoded
                else:
                    negative_prompt_embeds = encoded
                    negative_pooled_prompt_embeds = None

                encoded_negative_prompt_embeds.append(negative_prompt_embeds)
                if negative_pooled_prompt_embeds is not None:
                    encoded_negative_pooled_prompt_embeds.append(negative_pooled_prompt_embeds)

            if is_offloaded:
                logger.debug(f"Returning offloaded text encoder {i+1} to CPU.")
                text_encoders[i].to("cpu")

        use_last_pooled_embed = self.model_type == "sdxl"
        stack_dim = -1 if self.model_type == "sdxl" else -2

        if stack_dim not in [-1, encoded_prompt_embeds[0].ndim-1]:
            # Pad to the longest prompt
            longest_prompt_embed = max(embed.shape[-1] for embed in encoded_prompt_embeds)
            for i, prompt_embed in enumerate(encoded_prompt_embeds):
                if prompt_embed.shape[-1] < longest_prompt_embed:
                    encoded_prompt_embeds[i] = torch.nn.functional.pad(
                        prompt_embed,
                        (0, longest_prompt_embed - prompt_embed.shape[-1]),
                    )

        prompt_embeds = torch.cat(encoded_prompt_embeds, dim=stack_dim)
        if num_negative_prompts > 0:
            if stack_dim not in [-1, encoded_negative_prompt_embeds[0].ndim-1]:
                # Pad to the longest prompt
                for i, negative_prompt_embed in enumerate(encoded_negative_prompt_embeds):
                    if negative_prompt_embed.shape[-1] < longest_prompt_embed:
                        encoded_negative_prompt_embeds[i] = torch.nn.functional.pad(
                            negative_prompt_embed,
                            (0, longest_prompt_embed - negative_prompt_embed.shape[-1]),
                        )

            negative_prompt_embeds = torch.cat(encoded_negative_prompt_embeds, dim=stack_dim)
        else:
            negative_prompt_embeds = None # type: ignore[assignment]

        if encoded_pooled_prompt_embeds:
            if use_last_pooled_embed:
                encoded_pooled_prompt_embeds = encoded_pooled_prompt_embeds[-1] # type: ignore[assignment]
            else:
                encoded_pooled_prompt_embeds = torch.cat(encoded_pooled_prompt_embeds, dim=-1) # type: ignore[assignment]
        else:
            encoded_pooled_prompt_embeds = None # type: ignore[assignment]

        if encoded_negative_pooled_prompt_embeds:
            if use_last_pooled_embed:
                encoded_negative_pooled_prompt_embeds = encoded_negative_pooled_prompt_embeds[-1] # type: ignore[assignment]
            else:
                encoded_negative_pooled_prompt_embeds = torch.cat(encoded_negative_pooled_prompt_embeds, dim=-1) # type: ignore[assignment]
        else:
            encoded_negative_pooled_prompt_embeds = None # type: ignore[assignment]

        return prompt_embeds, encoded_pooled_prompt_embeds, negative_prompt_embeds, encoded_negative_pooled_prompt_embeds # type: ignore[return-value]

    def compile_prompts_into_kwargs(
        self,
        pipeline: DiffusionPipeline,
        kwargs: Dict[str, Any],
        accepts_negative_prompt: bool,
        clip_skip: Optional[int]=None,
    ) -> None:
        """
        Compiles prompts using compel, updating the kwarg dictionary in-place.
        """
        prompts = self.get_prompts_from_kwargs(**kwargs)
        negative_prompts = None if not accepts_negative_prompt else self.get_negative_prompts_from_kwargs(**kwargs)

        compiled_prompt_embeds = self.get_compiled_prompt_embeds(
            pipeline,
            prompts=prompts,
            negative_prompts=negative_prompts,
            clip_skip=clip_skip,
        )
        if compiled_prompt_embeds is None:
            return

        (
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = compiled_prompt_embeds

        kwargs["prompt_embeds"] = prompt_embeds.to(self.device)
        if pooled_prompt_embeds is not None:
            kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds.to(self.device)
        if negative_prompt_embeds is not None:
            kwargs["negative_prompt_embeds"] = negative_prompt_embeds.to(self.device)
        if negative_pooled_prompt_embeds is not None:
            kwargs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds.to(self.device)

        for i in range(3): # max known TE's is 3
            if i == 0:
                kwargs.pop("prompt", None)
                kwargs.pop("negative_prompt", None)
            else:
                kwargs.pop(f"prompt_{i+1}", None)
                kwargs.pop(f"negative_prompt_{i+1}", None)

    def get_encoded_spatial_prompts(
        self,
        pipeline: DiffusionPipeline,
        kwargs: Dict[str, Any],
        accepts_negative_prompt: bool,
        clip_skip: Optional[int]=None,
        spatial_prompts: Optional[List[SpatioTemporalPrompt]]=None,
    ) -> EncodedPrompts:
        """
        Get encoded spatial prompts.
        """
        # Instantiate holder
        encoded_prompts = EncodedPrompts()

        # Add the spatial prompts
        if spatial_prompts is not None:
            for spatial_prompt in maybe_use_tqdm(spatial_prompts, desc="Encoding spatial prompts"):
                prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds = self.get_compiled_prompt_embeds( # type: ignore[misc]
                    pipeline,
                    prompts=[spatial_prompt.prompt],
                    negative_prompts=None if not accepts_negative_prompt or not spatial_prompt.negative_prompt else [spatial_prompt.negative_prompt],
                    clip_skip=clip_skip,
                )
                encoded_prompt = EncodedPrompt(
                    embeddings=prompt_embeds,
                    pooled_embeddings=pooled_prompt_embeds,
                    negative_embeddings=negative_prompt_embeds,
                    negative_pooled_embeddings=negative_pooled_prompt_embeds,
                    position=spatial_prompt.position,
                    weight=spatial_prompt.weight,
                )
                encoded_prompts.add_prompt(encoded_prompt)

        # Add the base prompts (already encoded)
        if kwargs.get("prompt_embeds", None) is not None:
            base_prompt_embeds = kwargs["prompt_embeds"].clone().cpu()
            base_pooled_embeds = kwargs.get("pooled_prompt_embeds", None)
            if base_pooled_embeds is not None:
                base_pooled_embeds = base_pooled_embeds.clone().cpu()
            base_negative_embeds = kwargs.get("negative_prompt_embeds", None)
            if base_negative_embeds is not None:
                base_negative_embeds = base_negative_embeds.clone().cpu()
            base_negative_pooled_embeds = kwargs.get("negative_pooled_prompt_embeds", None)
            if base_negative_pooled_embeds is not None:
                base_negative_pooled_embeds = base_negative_pooled_embeds.clone().cpu()
            base_prompt = EncodedPrompt(
                embeddings=base_prompt_embeds,
                pooled_embeddings=base_pooled_embeds,
                negative_embeddings=base_negative_embeds,
                negative_pooled_embeddings=base_negative_pooled_embeds,
                weight=GLOBAL_PROMPT_WEIGHT,
            )
            encoded_prompts.add_prompt(base_prompt)
        return encoded_prompts

    def get_spatial_prompts(
        self,
        spatial_prompts: SpatialPromptInputType,
        add_default_negative_prompt: bool=True,
    ) -> List[SpatioTemporalPrompt]:
        """
        Gets formatted spatial prompts.
        """
        if isinstance(spatial_prompts, str):
            if re.search(r"\[.*\]", spatial_prompts) or re.search(r"\{.*\}", spatial_prompts):
                spatial_prompts = json.loads(spatial_prompts)
            elif os.path.exists(spatial_prompts):
                with open(spatial_prompts, "r") as f:
                    spatial_prompts = json.load(f)
        if not isinstance(spatial_prompts, (tuple, list)):
            spatial_prompts = [spatial_prompts] # type: ignore[list-item]

        prompts: List[SpatioTemporalPrompt] = []
        for prompt in spatial_prompts:
            if isinstance(prompt, str):
                prompts.append(
                    SpatioTemporalPrompt(
                        prompt=prompt,
                        negative_prompt=self.default_negative_prompt if add_default_negative_prompt else None
                    )
                )
            elif isinstance(prompt, dict):
                if prompt.get("negative_prompt", None) is None and add_default_negative_prompt:
                    prompt["negative_prompt"] = self.default_negative_prompt
                prompts.append(SpatioTemporalPrompt(**prompt))
            elif isinstance(prompt, SpatioTemporalPrompt):
                if prompt.negative_prompt is None and add_default_negative_prompt:
                    prompt.negative_prompt = self.default_negative_prompt
                prompts.append(prompt)
            else:
                raise ValueError(f"Invalid spatial prompt: {prompt}")

        return prompts

    def get_generator(self, seed: Optional[SeedType]=None) -> torch.Generator:
        """
        Get the generator.
        """
        import torch
        generator = torch.Generator(device=self.device)
        generator.manual_seed(get_seed(seed))
        return generator
