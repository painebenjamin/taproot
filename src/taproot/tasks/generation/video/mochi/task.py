from __future__ import annotations

from typing import Optional, Union, List, Type, Dict, TYPE_CHECKING

from taproot.constants import *
from taproot.util import log_duration, is_multiple
from taproot.pretrained import (
    T5XXLTextEncoder,
    T5XXLTextEncoderInt8,
    T5XXLTextEncoderNF4,
    T5XXLTokenizer,
)

from ..base import DiffusersTextToVideoTask
from .pretrained import (
    MochiVAE,
    MochiScheduler,
    MochiTransformer,
    MochiTransformerInt8,
    MochiTransformerNF4,
)

if TYPE_CHECKING:
    import torch
    from diffusers.pipelines import DiffusionPipeline
    from taproot.hinting import SeedType, ImageResultType

__all__ = [
    "MochiVideoGeneration",
    "MochiVideoGenerationInt8",
    "MochiVideoGenerationNF4",
]

class MochiVideoGeneration(DiffusersTextToVideoTask):
    """
    Text-to-video generation using Mochi-V1
    """
    
    """Global Task Metadata"""
    task = "video-generation"
    model = "mochi-v1"
    default = False
    display_name = "Mochi Video Generation"

    """Model Configuration"""
    pretrained_models = {
        "tokenizer": T5XXLTokenizer,
        "text_encoder": T5XXLTextEncoder,
        "transformer": MochiTransformer,
        "scheduler": MochiScheduler,
        "vae": MochiVAE,
    }
    gpu_precision: str = "bfloat16"
    static_gpu_memory_gb = 22.95
    static_memory_gb = 0.15646

    """Authorship Metadata"""
    author = "Genmo AI"
    author_url = "https://www.genmo.ai/blog"
    author_journal = "Genmo AI Blog"
    author_journal_year = 2024
    author_journal_title = "Mochi 1: A new SOTA in open-source video generation models"

    """Licensing Metadata"""
    licen = LICENSE_APACHE

    @classmethod
    def get_text_to_video_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Gets the pipeline class.
        """
        from diffusers.pipelines import MochiPipeline
        return MochiPipeline # type: ignore[return-value]

    def get_pipeline_modules(self) -> Dict[str, torch.nn.Module]:
        """
        Return the modules used in the pipeline.
        """
        return {
            "vae": self.pretrained.vae,
            "transformer": self.pretrained.transformer,
            "scheduler": self.pretrained.scheduler,
            "tokenizer": self.pretrained.tokenizer,
            "text_encoder": self.pretrained.text_encoder,
        }

    def __call__( # type: ignore[override]
        self,
        *,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]]=None,
        height: int=480,
        width: int=848,
        num_frames: int=19,
        num_inference_steps: int=50,
        timesteps: Optional[List[int]]=None,
        guidance_scale: float=4.5,
        num_videos_per_prompt: Optional[int]=1,
        frame_rate: int=8, # Mochi uses this as conditioning
        latents: Optional[torch.Tensor]=None,
        prompt_embeds: Optional[torch.Tensor]=None,
        prompt_attention_mask: Optional[torch.Tensor]=None,
        negative_prompt_embeds: Optional[torch.Tensor]=None,
        negative_prompt_attention_mask: Optional[torch.Tensor]=None,
        max_sequence_length: int=256,
        seed: Optional[SeedType]=None,
        output_format: VIDEO_OUTPUT_FORMAT_LITERAL="mp4",
        output_upload: bool=False,
    ) -> ImageResultType:
        """
        Generate a video from a text prompt.
        """
        with log_duration("inference"):
            results = self.invoke_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                num_videos_per_prompt=num_videos_per_prompt,
                frame_rate=frame_rate,
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                max_sequence_length=max_sequence_length,
                seed=seed,
            )

        is_single_video = num_videos_per_prompt == 1 and not is_multiple(prompt)
        return self.get_output_from_video_result(
            results,
            multi_video=not is_single_video,
            frame_rate=frame_rate,
            output_format=output_format,
            output_upload=output_upload,
        )

class MochiVideoGenerationInt8(MochiVideoGeneration):
    """
    Mochi Video Generation with int8 quantization on T5XXLTextEncoder and MochiTransformer.
    """
    model = "mochi-v1-int8"
    default = False
    static_gpu_memory_gb = 15.95
    pretrained_models = {
        **MochiVideoGeneration.pretrained_models,
        **{
            "text_encoder": T5XXLTextEncoderInt8,
            "transformer": MochiTransformerInt8,
        },
    }

class MochiVideoGenerationNF4(MochiVideoGeneration):
    """
    Mochi Video Generation with nf4 quantization on T5XXLTextEncoder and MochiTransformer.
    """
    model = "mochi-v1-nf4"
    default = False
    static_gpu_memory_gb = 12.41
    pretrained_models = {
        **MochiVideoGeneration.pretrained_models,
        **{
            "text_encoder": T5XXLTextEncoderNF4,
            "transformer": MochiTransformerNF4,
        },
    }
