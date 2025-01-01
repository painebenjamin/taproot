from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel

__all__ = [
    "CogVideoXTransformer2B",
    "CogVideoXTransformer2BInt8",
    "CogVideoXTransformer5B",
    "CogVideoXTransformer5BInt8",
    "CogVideoXTransformer5BNF4",
    "CogVideoXTransformerI2V5B",
    "CogVideoXTransformerI2V5BInt8",
    "CogVideoXTransformerI2V5BNF4",
    "CogVideoX15Transformer5B",
    "CogVideoX15Transformer5BInt8",
    "CogVideoX15Transformer5BNF4",
    "CogVideoX15TransformerI2V5B",
    "CogVideoX15TransformerI2V5BInt8",
    "CogVideoX15TransformerI2V5BNF4"
]

class CogVideoXTransformer(PretrainedModelMixin):
    """
    CogVideoX Transformer model base
    """
    @classmethod
    def get_model_class(cls) -> Type[CogVideoXTransformer3DModel]:
        """
        Returns the model class.
        """
        from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
        return CogVideoXTransformer3DModel # type: ignore[no-any-return]

class CogVideoXTransformer2B(CogVideoXTransformer):
    """
    CogVideoX 2B Transformer model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-transformer-2b.fp16.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "activation_fn": "gelu-approximate",
            "attention_bias": True,
            "attention_head_dim": 64,
            "dropout": 0.0,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 16,
            "max_text_seq_length": 226,
            "norm_elementwise_affine": True,
            "norm_eps": 1e-05,
            "num_attention_heads": 30,
            "num_layers": 30,
            "out_channels": 16,
            "patch_size": 2,
            "sample_frames": 49,
            "sample_height": 60,
            "sample_width": 90,
            "spatial_interpolation_scale": 1.875,
            "temporal_compression_ratio": 4,
            "temporal_interpolation_scale": 1.0,
            "text_embed_dim": 4096,
            "time_embed_dim": 512,
            "timestep_activation_fn": "silu",
            "use_rotary_positional_embeddings": False
        }

class CogVideoXTransformer2BInt8(CogVideoXTransformer2B):
    """
    CogVideoX 2B Transformer model with int8 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-transformer-2b.int8.fp16.safetensors"
    quantization = "bitsandbytes_8bit"
    pre_quantized = True

class CogVideoXTransformer5B(CogVideoXTransformer):
    """
    CogVideoX 5B Transformer model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-transformer-5b.fp16.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "activation_fn": "gelu-approximate",
            "attention_bias": True,
            "attention_head_dim": 64,
            "dropout": 0.0,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 16,
            "max_text_seq_length": 226,
            "norm_elementwise_affine": True,
            "norm_eps": 1e-05,
            "num_attention_heads": 48,
            "num_layers": 42,
            "out_channels": 16,
            "patch_size": 2,
            "sample_frames": 49,
            "sample_height": 60,
            "sample_width": 90,
            "spatial_interpolation_scale": 1.875,
            "temporal_compression_ratio": 4,
            "temporal_interpolation_scale": 1.0,
            "text_embed_dim": 4096,
            "time_embed_dim": 512,
            "timestep_activation_fn": "silu",
            "use_rotary_positional_embeddings": True
        }

class CogVideoXTransformer5BInt8(CogVideoXTransformer5B):
    """
    Cog VideoX 5B Transformer model with int8 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-transformer-5b.int8.fp16.safetensors"
    quantization = "bitsandbytes_8bit"
    pre_quantized = True

class CogVideoXTransformer5BNF4(CogVideoXTransformer5B):
    """
    Cog VideoX 5B Transformer model with NF4 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-transformer-5b.nf4.fp16.safetensors"
    quantization = "bitsandbytes_nf4"
    pre_quantized = True

class CogVideoXTransformerI2V5B(CogVideoXTransformer):
    """
    CogVideoX Image-to-Video 5B Transformer model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-i2v-transformer-5b.fp16.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "activation_fn": "gelu-approximate",
            "attention_bias": True,
            "attention_head_dim": 64,
            "dropout": 0.0,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 32,
            "max_text_seq_length": 226,
            "norm_elementwise_affine": True,
            "norm_eps": 1e-05,
            "num_attention_heads": 48,
            "num_layers": 42,
            "out_channels": 16,
            "patch_size": 2,
            "sample_frames": 49,
            "sample_height": 60,
            "sample_width": 90,
            "spatial_interpolation_scale": 1.875,
            "temporal_compression_ratio": 4,
            "temporal_interpolation_scale": 1.0,
            "text_embed_dim": 4096,
            "time_embed_dim": 512,
            "timestep_activation_fn": "silu",
            "use_learned_positional_embeddings": True,
            "use_rotary_positional_embeddings": True
        }

class CogVideoXTransformerI2V5BInt8(CogVideoXTransformerI2V5B):
    """
    CogVideoX Image-to-Video 5B Transformer model with int8 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-i2v-transformer-5b.fp16.safetensors"
    quantization = "bitsandbytes_8bit"
    pre_quantized = True

class CogVideoXTransformerI2V5BNF4(CogVideoXTransformerI2V5B):
    """
    CogVideoX Image-to-Video 5B Transformer model with NF4 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-i2v-transformer-5b.nf4.fp16.safetensors"
    quantization = "bitsandbytes_nf4"
    pre_quantized = True

class CogVideoX15Transformer5B(CogVideoXTransformer):
    """
    CogVideoX 1.5 5B Transformer model
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-v1-5-transformer-5b.fp16.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "activation_fn": "gelu-approximate",
            "attention_bias": True,
            "attention_head_dim": 64,
            "dropout": 0.0,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 16,
            "max_text_seq_length": 226,
            "norm_elementwise_affine": True,
            "norm_eps": 1e-05,
            "num_attention_heads": 48,
            "num_layers": 42,
            "out_channels": 16,
            "patch_bias": False,
            "patch_size": 2,
            "patch_size_t": 2,
            "sample_frames": 81,
            "sample_height": 300,
            "sample_width": 300,
            "spatial_interpolation_scale": 1.875,
            "temporal_compression_ratio": 4,
            "temporal_interpolation_scale": 1.0,
            "text_embed_dim": 4096,
            "time_embed_dim": 512,
            "timestep_activation_fn": "silu",
            "use_learned_positional_embeddings": False,
            "use_rotary_positional_embeddings": True
        }

class CogVideoX15Transformer5BInt8(CogVideoX15Transformer5B):
    """
    CogVideoX 1.5 5B Transformer model with int8 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-v1-5-transformer-5b.int8.fp16.safetensors"
    quantization = "bitsandbytes_8bit"
    pre_quantized = True

class CogVideoX15Transformer5BNF4(CogVideoX15Transformer5B):
    """
    CogVideoX 1.5 5B Transformer model with NF4 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-v1-5-transformer-5b.nf4.fp16.safetensors"
    quantization = "bitsandbytes_nf4"
    pre_quantized = True

class CogVideoX15TransformerI2V5B(CogVideoXTransformer):
    """
    CogVideoX 1.5 Image-to-Video 5B Transformer model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-v1-5-i2v-transformer-5b.fp16.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "activation_fn": "gelu-approximate",
            "attention_bias": True,
            "attention_head_dim": 64,
            "dropout": 0.0,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 32,
            "max_text_seq_length": 226,
            "norm_elementwise_affine": True,
            "norm_eps": 1e-05,
            "num_attention_heads": 48,
            "num_layers": 42,
            "ofs_embed_dim": 512,
            "out_channels": 16,
            "patch_bias": False,
            "patch_size": 2,
            "patch_size_t": 2,
            "sample_frames": 81,
            "sample_height": 300,
            "sample_width": 300,
            "spatial_interpolation_scale": 1.875,
            "temporal_compression_ratio": 4,
            "temporal_interpolation_scale": 1.0,
            "text_embed_dim": 4096,
            "time_embed_dim": 512,
            "timestep_activation_fn": "silu",
            "use_learned_positional_embeddings": False,
            "use_rotary_positional_embeddings": True
        }

class CogVideoX15TransformerI2V5BInt8(CogVideoX15TransformerI2V5B):
    """
    CogVideoX 1.5 Image-to-Video 5B Transformer model with int8 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-v1-5-i2v-transformer-5b.int8.fp16.safetensors"
    quantization = "bitsandbytes_8bit"
    pre_quantized = True

class CogVideoX15TransformerI2V5BNF4(CogVideoX15TransformerI2V5B):
    """
    CogVideoX 1.5 Image-to-Video 5B Transformer model with NF4 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/video-generation-cog-v1-5-i2v-transformer-5b.nf4.fp16.safetensors"
    quantization = "bitsandbytes_nf4"
    pre_quantized = True
