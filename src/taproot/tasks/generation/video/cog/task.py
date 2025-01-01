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
    CogVideoXVAEBF16,
    CogVideoXVAE5BBF16,
    CogVideoX15VAE5BBF16,
    CogVideoX2BScheduler,
    CogVideoX5BScheduler,
    CogVideoXTransformer2B,
    CogVideoXTransformer2BInt8,
    CogVideoXTransformer5B,
    CogVideoXTransformer5BInt8,
    CogVideoXTransformer5BNF4,
    CogVideoXTransformerI2V5B,
    CogVideoXTransformerI2V5BInt8,
    CogVideoXTransformerI2V5BNF4,
    CogVideoX15Transformer5B,
    CogVideoX15Transformer5BInt8,
    CogVideoX15Transformer5BNF4,
    CogVideoX15TransformerI2V5B,
    CogVideoX15TransformerI2V5BInt8,
    CogVideoX15TransformerI2V5BNF4,
)

if TYPE_CHECKING:
    import torch
    from diffusers.pipelines import DiffusionPipeline
    from taproot.hinting import SeedType, ImageResultType, ImageType

__all__ = [
    "CogVideoX2BVideoGeneration",
    "CogVideoX2BVideoGenerationInt8",
    "CogVideoX5BVideoGeneration",
    "CogVideoX5BVideoGenerationInt8",
    "CogVideoX5BVideoGenerationNF4",
    "CogVideoXI2V5BVideoGeneration",
    "CogVideoXI2V5BVideoGenerationInt8",
    "CogVideoXI2V5BVideoGenerationNF4",
    "CogVideoX155BVideoGeneration",
    "CogVideoX155BVideoGenerationInt8",
    "CogVideoX155BVideoGenerationNF4",
    "CogVideoX15I2V5BVideoGeneration",
    "CogVideoX15I2V5BVideoGenerationInt8",
    "CogVideoX15I2V5BVideoGenerationNF4",
]

class CogVideoX2BVideoGeneration(DiffusersTextToVideoTask):
    """
    Text-to-video generation using CogVideoX.
    """
    
    """Global Task Metadata"""
    task = "video-generation"
    model = "cogvideox-2b"
    default = False
    display_name = "CogVideoX 2B Video Generation"

    """Model Configuration"""
    pretrained_models = {
        "tokenizer": T5XXLTokenizer,
        "text_encoder": T5XXLTextEncoder,
        "transformer": CogVideoXTransformer2B,
        "scheduler": CogVideoX2BScheduler,
        "vae": CogVideoXVAEBF16,
    }
    static_gpu_memory_gb = 13.48
    static_memory_gb = 0.15646
    gpu_precision: str = "float16"
    default_negative_prompt = None

    """Authorship Metadata"""
    author = "Zhuoyi Yang"
    author_additional = [
        "Jiayen Teng", "Wendi Zheng", "Ming Ding", "Shiyu Huang", "Jiazheng Xu", "Yuanming Yang", "Wenyi Hong", "Xiaohan Zhang", "Guanyu Feng", "Da Yin", "Xiaotao Gu", "Yuxuan Zhang", "Weihan Wang", "Yean Cheng", "Ting Liu", "Bin Xu", "Yuxiao Dong", "Jie Tang"
    ]
    author_affiliations = ["Zhipu AI", "Tsinghua University"]
    author_journal = "arXiv"
    author_journal_title = "CogVideoX: Text-to-Video Diffusion Models with an Experty Transformer"
    author_journal_volume = "2408.06072"
    author_journal_year = 2024
    author_url = "https://arxiv.org/abs/2408.06072"

    """License Metadata"""
    license = "CogVideoX License"
    license_url = "https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE"
    license_attribution = True
    license_derivatives = True
    license_redistribution = True
    license_copy_left = False
    license_commercial = True # Limited at $1M revenue but not before
    license_hosting = True

    @classmethod
    def get_text_to_video_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Gets the pipeline class.
        """
        from diffusers.pipelines import CogVideoXPipeline
        return CogVideoXPipeline # type: ignore[return-value]

    @classmethod
    def get_image_to_video_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Gets the pipeline class.
        """
        from diffusers.pipelines import CogVideoXImageToVideoPipeline
        return CogVideoXImageToVideoPipeline # type: ignore[return-value]

    @classmethod
    def get_video_to_video_pipeline_class(cls) -> Type[DiffusionPipeline]:
        """
        Gets the pipeline class.
        """
        from diffusers.pipelines import CogVideoXVideoToVideoPipeline
        return CogVideoXVideoToVideoPipeline # type: ignore[return-value]

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
        height: int=480,
        width: int=720,
        num_frames: int=49,
        num_inference_steps: int=50,
        timesteps: Optional[List[int]]=None,
        guidance_scale: float=6.0,
        num_videos_per_prompt: Optional[int]=1,
        latents: Optional[torch.Tensor]=None,
        prompt_embeds: Optional[torch.Tensor]=None,
        negative_prompt_embeds: Optional[torch.Tensor]=None,
        max_sequence_length: int=226,
        seed: Optional[SeedType]=None,
        output_format: VIDEO_OUTPUT_FORMAT_LITERAL="mp4",
        output_upload: bool=False,
        frame_rate: int=8,
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
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
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

class CogVideoX2BVideoGenerationInt8(CogVideoX2BVideoGeneration):
    """
    Text-to-video generation using CogVideoX-2B with int8 quantization.
    """
    
    """Global Task Metadata"""
    model = "cogvideox-2b-int8"
    display_name = "CogVideoX 2B Video Generation (Int8)"

    """Model Configuration"""
    static_gpu_memory_gb = 11.48
    pretrained_models = {
        **CogVideoX2BVideoGeneration.pretrained_models,
        **{
            "text_encoder": T5XXLTextEncoderInt8,
            "transformer": CogVideoXTransformer2BInt8,
        }
    }

class CogVideoX5BVideoGeneration(CogVideoX2BVideoGeneration):
    """
    Text-to-video generation using CogVideoX-5B.
    """
    
    """Global Task Metadata"""
    task = "video-generation"
    model = "cogvideox-5b"
    default = False
    display_name = "CogVideoX 5B Video Generation"

    """Model Configuration"""
    pretrained_models = {
        **CogVideoX2BVideoGeneration.pretrained_models,
        **{
            "transformer": CogVideoXTransformer5B,
            "scheduler": CogVideoX5BScheduler,
            "vae": CogVideoXVAE5BBF16,
        }
    }
    static_gpu_memory_gb = 21.48
    gpu_precision: str = "bfloat16"

class CogVideoX5BVideoGenerationInt8(CogVideoX5BVideoGeneration):
    """
    Text-to-video generation using CogVideoX-5B with int8 quantization.
    """

    """Global Task Metadata"""
    model = "cogvideox-5b-int8"
    display_name = "CogVideoX 5B Video Generation (Int8)"

    """Model Configuration"""
    static_gpu_memory_gb = 17.48
    pretrained_models = {
        **CogVideoX5BVideoGeneration.pretrained_models,
        **{
            "text_encoder": T5XXLTextEncoderInt8,
            "transformer": CogVideoXTransformer5BInt8,
        }
    }

class CogVideoX5BVideoGenerationNF4(CogVideoX5BVideoGeneration):
    """
    Text-to-video generation using CogVideoX-5B with NF4 quantization.
    """

    """Global Task Metadata"""
    model = "cogvideox-5b-nf4"
    display_name = "CogVideoX 5B Video Generation (NF4)"

    """Model Configuration"""
    static_gpu_memory_gb = 12.48
    pretrained_models = {
        **CogVideoX5BVideoGeneration.pretrained_models,
        **{
            "text_encoder": T5XXLTextEncoderNF4,
            "transformer": CogVideoXTransformer5BNF4,
        }
    }

class CogVideoXI2V5BVideoGeneration(CogVideoX5BVideoGeneration):
    """
    Image-to-video generation using CogVideoX-5B.
    """

    """Global Task Metadata"""
    model = "cogvideox-i2v-5b"
    display_name = "CogVideoX 5B Image-to-Video Generation"

    """Model Configuration"""
    pretrained_models = {
        **CogVideoX5BVideoGeneration.pretrained_models,
        **{
            "transformer": CogVideoXTransformerI2V5B,
        }
    }

class CogVideoXI2V5BVideoGenerationInt8(CogVideoXI2V5BVideoGeneration):
    """
    Image-to-video generation using CogVideoX-5B with int8 quantization.
    """

    """Global Task Metadata"""
    model = "cogvideox-i2v-5b-int8"
    display_name = "CogVideoX 5B Image-to-Video Generation (Int8)"

    """Model Configuration"""
    static_gpu_memory_gb = 17.48
    pretrained_models = {
        **CogVideoXI2V5BVideoGeneration.pretrained_models,
        **{
            "transformer": CogVideoXTransformerI2V5BInt8,
            "text_encoder": T5XXLTextEncoderInt8,
        }
    }

class CogVideoXI2V5BVideoGenerationNF4(CogVideoXI2V5BVideoGeneration):
    """
    Image-to-video generation using CogVideoX-5B with NF4 quantization.
    """

    """Global Task Metadata"""
    model = "cogvideox-i2v-5b-nf4"
    display_name = "CogVideoX 5B Image-to-Video Generation (NF4)"

    """Model Configuration"""
    static_gpu_memory_gb = 12.48
    pretrained_models = {
        **CogVideoXI2V5BVideoGeneration.pretrained_models,
        **{
            "transformer": CogVideoXTransformerI2V5BNF4,
            "text_encoder": T5XXLTextEncoderNF4,
        }
    }

class CogVideoX155BVideoGeneration(CogVideoX5BVideoGeneration):
    """
    Text-to-video generation using CogVideoX V1.5 5B.
    """

    """Global Task Metadata"""
    model = "cogvideox-v1-5-5b"
    display_name = "CogVideoX V1.5 5B Video Generation"

    """Model Configuration"""
    static_gpu_memory_gb = 21.48
    pretrained_models = {
        **CogVideoX5BVideoGeneration.pretrained_models,
        **{
            "transformer": CogVideoX15Transformer5B,
            "vae": CogVideoX15VAE5BBF16,
        }
    }

    def __call__( # type: ignore[override]
        self,
        *,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]]=None,
        image: Optional[ImageType]=None,
        height: int=768,
        width: int=1360,
        num_frames: int=81,
        num_inference_steps: int=50,
        timesteps: Optional[List[int]]=None,
        guidance_scale: float=6.0,
        num_videos_per_prompt: Optional[int]=1,
        latents: Optional[torch.Tensor]=None,
        prompt_embeds: Optional[torch.Tensor]=None,
        negative_prompt_embeds: Optional[torch.Tensor]=None,
        max_sequence_length: int=224,
        seed: Optional[SeedType]=None,
        output_format: VIDEO_OUTPUT_FORMAT_LITERAL="mp4",
        output_upload: bool=False,
        frame_rate: int=8,
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
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
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

class CogVideoX155BVideoGenerationInt8(CogVideoX155BVideoGeneration):
    """
    Text-to-video generation using CogVideoX V1.5 5B Transformer and int8 quantization.
    """

    """Global Task Metadata"""
    model = "cogvideox-v1-5-5b-int8"
    display_name = "CogVideoX V1.5 5B Video Generation (Int8)"

    """Model Configuration"""
    static_gpu_memory_gb = 17.48
    pretrained_models = {
        **CogVideoX155BVideoGeneration.pretrained_models,
        **{
            "transformer": CogVideoX15Transformer5BInt8,
            "text_encoder": T5XXLTextEncoderInt8,
        }
    }

class CogVideoX155BVideoGenerationNF4(CogVideoX155BVideoGeneration):
    """
    Text-to-video generation using CogVideoX V1.5 5B Transformer and NF4 quantization.
    """

    """Global Task Metadata"""
    model = "cogvideox-v1-5-5b-nf4"
    display_name = "CogVideoX V1.5 5B Video Generation (NF4)"

    """Model Configuration"""
    static_gpu_memory_gb = 12.48
    pretrained_models = {
        **CogVideoX155BVideoGeneration.pretrained_models,
        **{
            "transformer": CogVideoX15Transformer5BNF4,
            "text_encoder": T5XXLTextEncoderNF4,
        }
    }

class CogVideoX15I2V5BVideoGeneration(CogVideoX155BVideoGeneration):
    """
    Image-to-video generation using CogVideoX V1.5 I2V 5B.
    """

    """Global Task Metadata"""
    model = "cogvideox-v1-5-i2v-5b"
    display_name = "CogVideoX V1.5 5B Image-to-Video Generation"

    """Model Configuration"""
    static_gpu_memory_gb = 21.48
    pretrained_models = {
        **CogVideoX155BVideoGeneration.pretrained_models,
        **{
            "transformer": CogVideoX15TransformerI2V5B,
        }
    }

class CogVideoX15I2V5BVideoGenerationInt8(CogVideoX15I2V5BVideoGeneration):
    """
    Image-to-video generation using CogVideoX V1.5 I2V 5B with int8 quantization.
    """

    """Global Task Metadata"""
    model = "cogvideox-v1-5-i2v-5b-int8"
    display_name = "CogVideoX V1.5 5B Image-to-Video Generation (Int8)"

    """Model Configuration"""
    static_gpu_memory_gb = 17.48
    pretrained_models = {
        **CogVideoX155BVideoGeneration.pretrained_models,
        **{
            "transformer": CogVideoX15TransformerI2V5BInt8,
            "text_encoder": T5XXLTextEncoderInt8,
        }
    }

class CogVideoX15I2V5BVideoGenerationNF4(CogVideoX15I2V5BVideoGeneration):
    """
    Image-to-video generation using CogVideoX V1.5 I2V 5B with NF4 quantization.
    """

    """Global Task Metadata"""
    model = "cogvideox-v1-5-i2v-5b-nf4"
    display_name = "CogVideoX V1.5 5B Image-to-Video Generation (NF4)"

    """Model Configuration"""
    static_gpu_memory_gb = 12.48
    pretrained_models = {
        **CogVideoX155BVideoGeneration.pretrained_models,
        **{
            "transformer": CogVideoX15TransformerI2V5BNF4,
            "text_encoder": T5XXLTextEncoderNF4,
        }
    }
