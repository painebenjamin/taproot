from __future__ import annotations

from taproot.util import PretrainedModelMixin

from typing import Optional, Type, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import ( # type: ignore[import-untyped]
        LlamaModel,
        LlamaConfig
    )

__all__ = [
    "LlavaLlamaTextEncoder",
    "LlavaLlamaTextEncoderInt8",
    "LlavaLlamaTextEncoderNF4"
]

class LlavaLlamaTextEncoder(PretrainedModelMixin):
    """
    The text encoder model for LLaVA-LLaMA, also used for
    HunyuanVideo and potentially others.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-llava-llama-text-encoder.fp16.safetensors"
    dtype = "float16"

    @classmethod
    def get_model_class(cls) -> Type[LlamaModel]:
        """
        Gets the model class.
        """
        from transformers import LlamaModel
        return LlamaModel # type: ignore[no-any-return]

    @classmethod
    def get_config_class(cls) -> Type[LlamaConfig]:
        """
        Gets the config class.
        """
        from transformers import LlamaConfig
        return LlamaConfig # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Gets the default configuration.
        """
        return {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": 8192,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "rope_theta": 500000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "use_cache": True,
            "vocab_size": 128320
        }

class LlavaLlamaTextEncoderInt8(LlavaLlamaTextEncoder):
    """
    The text encoder model for LLaVA-LLaMA with int8 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-llava-llama-text-encoder.int8.fp16.safetensors"
    quantization = "bitsandbytes_8bit"
    pre_quantized = True

class LlavaLlamaTextEncoderNF4(LlavaLlamaTextEncoder):
    """
    The text encoder model for LLaVA-LLaMA with nf4 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-llava-llama-text-encoder.nf4.fp16.safetensors"
    quantization = "bitsandbytes_nf4"
    pre_quantized = True
