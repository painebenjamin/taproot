from __future__ import annotations

from typing import Type, Optional, Dict, Any, Union, List, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

__all__ = [
    "StableDiffusion3Transformer",
    "StableDiffusion35MediumTransformer",
    "StableDiffusion35LargeTransformer",
    "StableDiffusion35LargeTransformerFP8",
    "StableDiffusion35LargeTransformerInt8",
    "StableDiffusion35LargeTransformerNF4"
]

class StableDiffusion3Transformer(PretrainedModelMixin):
    """
    Stable Diffusion V3 Transformer Model
    """
    model_url: Optional[Union[str, List[str]]] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v3-transformer.fp16.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[SD3Transformer2DModel]:
        """
        Get the model class for the Stable Diffusion XL Transformer.
        """
        from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
        return SD3Transformer2DModel # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Stable Diffusion XL Transformer.
        """
        return {
          "attention_head_dim": 64,
          "caption_projection_dim": 1536,
          "in_channels": 16,
          "joint_attention_dim": 4096,
          "num_attention_heads": 24,
          "num_layers": 24,
          "out_channels": 16,
          "patch_size": 2,
          "pooled_projection_dim": 2048,
          "pos_embed_max_size": 192,
          "sample_size": 128
        }

class StableDiffusion35MediumTransformer(StableDiffusion3Transformer):
    """
    StableDiffusion v3.5 Transformer Model (Medium)
    """
    model_url: Optional[Union[str, List[str]]] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v3-5-medium-transformer.bf16.safetensors"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get the default configuration for this model.
        """
        return {
            "dual_attention_layers": list(range(13)),
            "attention_head_dim": 64,
            "caption_projection_dim": 1536,
            "in_channels": 16,
            "joint_attention_dim": 4096,
            "num_attention_heads": 24,
            "num_layers": 24,
            "out_channels": 16,
            "patch_size": 2,
            "pooled_projection_dim": 2048,
            "pos_embed_max_size": 384,
            "qk_norm": "rms_norm",
            "sample_size": 128
        }

class StableDiffusion35LargeTransformer(StableDiffusion3Transformer):
    """
    StableDiffusion v3.5 Transformer Model (Large)
    """
    model_url: Optional[Union[str, List[str]]] = [
        "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v3-5-large-transformer.part-1.bf16.safetensors",
        "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v3-5-large-transformer.part-2.bf16.safetensors",
    ]

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get the default configuration for this model.
        """
        return {
            "attention_head_dim": 64,
            "caption_projection_dim": 2432,
            "in_channels": 16,
            "joint_attention_dim": 4096,
            "num_attention_heads": 38,
            "num_layers": 38,
            "out_channels": 16,
            "patch_size": 2,
            "pooled_projection_dim": 2048,
            "pos_embed_max_size": 192,
            "qk_norm": "rms_norm",
            "sample_size": 128
        }

class StableDiffusion35LargeTransformerFP8(StableDiffusion35LargeTransformer):
    """
    StableDiffusion v3.5 Transformer Model (Large) in FP8
    """
    model_url: Optional[Union[str, List[str]]] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v3-5-large-transformer.fp8-e4m3-fn.safetensors"
    dtype = "float8_e4m3fn"

class StableDiffusion35LargeTransformerInt8(StableDiffusion35LargeTransformer):
    """
    StableDiffusion v3.5 Transformer Model (Large) with 8-bit quantization
    """
    model_url: Optional[Union[str, List[str]]] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v3-5-large-transformer.int8.bf16.safetensors"
    quantization = "bitsandbytes_8bit"
    pre_quantized = True

class StableDiffusion35LargeTransformerNF4(StableDiffusion35LargeTransformer):
    """
    StableDiffusion v3.5 Transformer Model (Large) with 4-bit normalized floating point quantization (NF4)
    """
    model_url: Optional[Union[str, List[str]]] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v3-5-large-transformer.nf4.bf16.safetensors"
    quantization = "bitsandbytes_nf4"
    pre_quantized = True
