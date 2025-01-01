from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

__all__ = [
    "FluxDevTransformer",
    "FluxDevTransformerFP8",
    "FluxDevTransformerInt8",
    "FluxDevTransformerNF4",
    "FluxSchnellTransformer",
    "FluxSchnellTransformerFP8",
    "FluxSchnellTransformerInt8",
    "FluxSchnellTransformerNF4",
]

class FluxTransformer(PretrainedModelMixin):
    """
    The base model for Flux Transformers.
    """
    use_guidance_embeds: bool = True

    @classmethod
    def get_model_class(cls) -> Type[FluxTransformer2DModel]:
        """
        Get the model class for the Flux Transformer.
        """
        from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
        return FluxTransformer2DModel # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Flux Transformer.
        """
        return {
          "attention_head_dim": 128,
          "guidance_embeds": cls.use_guidance_embeds,
          "in_channels": 64,
          "joint_attention_dim": 4096,
          "num_attention_heads": 24,
          "num_layers": 19,
          "num_single_layers": 38,
          "patch_size": 1,
          "pooled_projection_dim": 768
        }

class FluxDevTransformer(FluxTransformer):
    """
    The base transformer model for FLUX.1-dev, with unquantized weights.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-dev-transformer.bf16.safetensors"

class FluxDevTransformerFP8(FluxTransformer):
    """
    The base transformer model for FLUX.1-dev, with 8-bit floating point weights.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-dev-transformer.fp8-e4m3-fn.safetensors"
    dtype = "float8_e4m3fn"

class FluxDevTransformerInt8(FluxTransformer):
    """
    The base transformer model for FLUX.1-dev, with 8-bit integer quantized weights.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-dev-transformer.int8.bf16.safetensors"
    quantization = "bitsandbytes_8bit"
    pre_quantized = True

class FluxDevTransformerNF4(FluxTransformer):
    """
    The base transformer model for FLUX.1-dev, with 4-bit normalized float quantized weights.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-dev-transformer.nf4.bf16.safetensors"
    quantization = "bitsandbytes_nf4"
    pre_quantized = True

class FluxSchnellTransformer(FluxTransformer):
    """
    The base transformer model for FLUX.1-schnell, with unquantized weights.
    """
    use_guidance_embeds = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-schnell-transformer.bf16.safetensors"

class FluxSchnellTransformerFP8(FluxTransformer):
    """
    The base transformer model for FLUX.1-schnell, with 8-bit floating point weights.
    """
    use_guidance_embeds = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-schnell-transformer.fp8-e4m3-fn.safetensors"
    dtype = "float8_e4m3fn"

class FluxSchnellTransformerInt8(FluxTransformer):
    """
    The base transformer model for FLUX.1-schnell, with 8-bit integer quantized weights.
    """
    use_guidance_embeds = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-schnell-transformer.int8.bf16.safetensors"
    quantization = "bitsandbytes_8bit"
    pre_quantized = True

class FluxSchnellTransformerNF4(FluxTransformer):
    """
    The base transformer model for FLUX.1-schnell, with 4-bit normalized float quantized weights.
    """
    use_guidance_embeds = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-flux-v1-schnell-transformer.nf4.bf16.safetensors"
    quantization = "bitsandbytes_nf4"
    pre_quantized = True
