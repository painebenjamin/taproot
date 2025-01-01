from __future__ import annotations

from typing import Optional, Any, Union, List, Type, Dict, TYPE_CHECKING

from taproot.constants import *
from taproot.util import log_duration, is_multiple

from taproot.pretrained import (
    LlavaLlamaTextEncoder,
    LlavaLlamaTextEncoderInt8,
    LlavaLlamaTextEncoderNF4,
    LlavaLlamaTokenizer,
    CLIPViTLTextEncoder,
    CLIPViTLTokenizer,
)

from ..base import DiffusersTextToVideoTask
from .pretrained import (
    HunyuanVideoVAE,
    HunyuanVideoScheduler,
    HunyuanVideoTransformer,
    HunyuanVideoTransformerInt8,
    HunyuanVideoTransformerNF4,
)

if TYPE_CHECKING:
    import torch
    from taproot.hinting import SeedType, ImageResultType
    from diffusers.pipelines import DiffusionPipeline

__all__ = [
    "HunyuanVideoGeneration",
    "HunyuanVideoGenerationInt8",
    "HunyuanVideoGenerationNF4"
]

class HunyuanVideoGeneration(DiffusersTextToVideoTask):
    """
    Text-to-video generation using Hunyuan Video.
    """
    
    """Global Task Metadata"""
    task = "video-generation"
    model = "hunyuan"
    default = False
    display_name = "Hunyuan Video Generation"

    """Model Configuration"""
    gpu_precision = "bfloat16"
    static_gpu_memory_gb = 38.3 # Could be more, can't test
    pretrained_models = {
        "vae": HunyuanVideoVAE,
        "transformer": HunyuanVideoTransformer,
        "scheduler": HunyuanVideoScheduler,
        "tokenizer": LlavaLlamaTokenizer,
        "tokenizer_2": CLIPViTLTokenizer,
        "text_encoder": LlavaLlamaTextEncoder,
        "text_encoder_2": CLIPViTLTextEncoder,
    }

    """Authorship Metadata"""
    author = "Hunyuan Foundation Model Team"
    author_affiliations = ["Tencent"]
    author_url = "https://arxiv.org/abs/2412.03603"
    author_journal = "arXiv"
    author_journal_year = 2024
    author_journal_volume = "2412.03603"
    author_journal_title = "HunyuanVideo: A Systematic Framework for Large Video Generation Models"

    """License Metadata"""
    license = "Tencent Hunyuan Community License"
    license_url = "https://github.com/Tencent/HunyuanVideo/blob/main/LICENSE.txt"
    license_attribution = True # Must attribute the authors
    license_redistribution = True # Can redistribute
    license_derivatives = True # Can modify
    license_commercial = True # Can use for commercial purposes up to 100 million users/month
    license_hosting = True # Can host the model as a service
    license_copy_left = False # Derived works do not have to be open source

    @classmethod
    def get_text_to_video_pipeline_class(cls, **kwargs: Any) -> Type[DiffusionPipeline]:
        """
        Returns the pipeline class to use for this task.
        """
        from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import HunyuanVideoPipeline
        return HunyuanVideoPipeline # type: ignore[return-value]

    def get_pipeline_modules(self) -> Dict[str, torch.nn.Module]:
        """
        Return all modules required for the pipeline.
        """
        return {
            "vae": self.pretrained.vae,
            "transformer": self.pretrained.transformer,
            "scheduler": self.pretrained.scheduler,
            "tokenizer": self.pretrained.tokenizer,
            "tokenizer_2": self.pretrained.tokenizer_2,
            "text_encoder": self.pretrained.text_encoder,
            "text_encoder_2": self.pretrained.text_encoder_2,
        }

    def __call__( # type: ignore[override]
        self,
        *,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]]=None,
        height: int=720,
        width: int=1280,
        num_frames: int=129,
        num_inference_steps: int=50,
        sigmas: Optional[List[float]]=None,
        guidance_scale: float=6.0,
        num_videos_per_prompt: Optional[int]=1,
        latents: Optional[torch.Tensor]=None,
        prompt_embeds: Optional[torch.Tensor]=None,
        pooled_prompt_embeds: Optional[torch.Tensor]=None,
        prompt_attention_mask: Optional[torch.Tensor]=None,
        max_sequence_length: int=256,
        frame_rate: int=DEFAULT_FRAME_RATE,
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
                prompt_2=prompt_2,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_videos_per_prompt=num_videos_per_prompt,
                latents=latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
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

class HunyuanVideoGenerationInt8(HunyuanVideoGeneration):
    """
    Hunyuan Video Generation with INT8 quantization.
    """
    model = "hunyuan-int8"
    pretrained_models = {
        **HunyuanVideoGeneration.pretrained_models,
        **{
            "transformer": HunyuanVideoTransformerInt8,
            "text_encoder": LlavaLlamaTextEncoderInt8,
        }
    }
    static_gpu_memory_gb = 23.3 # Can load model but can't execute

class HunyuanVideoGenerationNF4(HunyuanVideoGeneration):
    """
    Hunyuan Video Generation with NF4 quantization.
    """
    model = "hunyuan-nf4"
    gpu_precision = "float32"
    pretrained_models = {
        **HunyuanVideoGeneration.pretrained_models,
        **{
            "transformer": HunyuanVideoTransformerNF4,
            "text_encoder": LlavaLlamaTextEncoderNF4,
        }
    }
    static_gpu_memory_gb = 14.78 # Can load and execute model
