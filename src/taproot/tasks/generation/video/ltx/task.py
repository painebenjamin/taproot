from __future__ import annotations

from typing import Optional, Union, List, Dict, Type, TYPE_CHECKING

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
    LTXVideoVAE,
    LTXVideoScheduler,
    LTXVideoTransformer,
    LTXVideoTransformerInt8,
    LTXVideoTransformerNF4,
)

if TYPE_CHECKING:
    import torch
    from diffusers.pipelines import DiffusionPipeline
    from taproot.hinting import SeedType, ImageResultType, ImageType

__all__ = [
    "LTXVideoGeneration",
    "LTXVideoGenerationInt8",
    "LTXVideoGenerationNF4",
]

class LTXVideoGeneration(DiffusersTextToVideoTask):
    """
    Text-to-video generation using LTX Video.
    """
    
    """Global Task Metadata"""
    task = "video-generation"
    model = "ltx"
    default = True
    display_name = "LTX Video Generation"

    """Model Configuration"""
    pretrained_models = {
        "tokenizer": T5XXLTokenizer,
        "text_encoder": T5XXLTextEncoder,
        "transformer": LTXVideoTransformer,
        "scheduler": LTXVideoScheduler,
        "vae": LTXVideoVAE,
    }
    gpu_precision: str = "bfloat16"
    static_gpu_memory_gb = 15.28
    static_memory_gb = 0.15646

    """Authorship Metadata"""
    author = "Lightricks"
    author_url = "https://github.com/Lightricks/LTX-Video"

    """Licensing Metadata"""
    license = LICENSE_OPENRAIL

    @classmethod
    def get_text_to_video_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Gets the pipeline class.
        """
        from diffusers.pipelines import LTXPipeline
        return LTXPipeline # type: ignore[return-value]

    @classmethod
    def get_image_to_video_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Gets the pipeline class.
        """
        from diffusers.pipelines import LTXImageToVideoPipeline
        return LTXImageToVideoPipeline # type: ignore[return-value]

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
        image: Optional[ImageType]=None,
        height: int=512,
        width: int=704,
        num_frames: int=161,
        num_inference_steps: int=50,
        timesteps: Optional[List[int]]=None,
        guidance_scale: float=3.0,
        num_videos_per_prompt: Optional[int]=1,
        frame_rate: int=25, # LTX uses this as conditioning
        latents: Optional[torch.Tensor]=None,
        prompt_embeds: Optional[torch.Tensor]=None,
        prompt_attention_mask: Optional[torch.Tensor]=None,
        negative_prompt_embeds: Optional[torch.Tensor]=None,
        negative_prompt_attention_mask: Optional[torch.Tensor]=None,
        decode_timestep: Union[float, List[float]]=0.0,
        decode_noise_scale: Optional[Union[float, List[float]]]=None,
        max_sequence_length: int=128,
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
                image=image,
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
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
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

class LTXVideoGenerationInt8(LTXVideoGeneration):
    """
    LTX Video Generation with int8 quantization on T5XXLTextEncoder and LTXVideoTransformer.
    """
    model = "ltx-int8"
    default = False
    static_gpu_memory_gb = 9.72
    pretrained_models = {
        **LTXVideoGeneration.pretrained_models,
        **{
            "text_encoder": T5XXLTextEncoderInt8,
            "transformer": LTXVideoTransformerInt8,
        },
    }

class LTXVideoGenerationNF4(LTXVideoGeneration):
    """
    LTX Video Generation with nf4 quantization on T5XXLTextEncoder and LTXVideoTransformer.
    """
    model = "ltx-nf4"
    default = False
    static_gpu_memory_gb = 7.29
    pretrained_models = {
        **LTXVideoGeneration.pretrained_models,
        **{
            "text_encoder": T5XXLTextEncoderNF4,
            "transformer": LTXVideoTransformerNF4,
        },
    }
