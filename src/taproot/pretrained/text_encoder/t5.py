from __future__ import annotations

from taproot.util import PretrainedModelMixin

from typing import Optional, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import ( # type: ignore[import-untyped]
        T5EncoderModel,
        T5Config,
    )

__all__ = [
    "T5XXLTextEncoder",
    "T5XXLTextEncoderInt8",
    "T5XXLTextEncoderNF4",
]

class T5XXLTextEncoder(PretrainedModelMixin):
    """
    T5-XXL Text Encoder model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-t5-xxl.bf16.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[T5EncoderModel]:
        """
        Get the model class for the T5-XXL Text Encoder.
        """
        from transformers import T5EncoderModel
        return T5EncoderModel # type: ignore[no-any-return]

    @classmethod
    def get_config_class(cls) -> Type[T5Config]:
        """
        Get the configuration class for the T5-XXL Text Encoder.
        """
        from transformers import T5Config
        return T5Config # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the T5-XXL Text Encoder.
        """
        return {
            "classifier_dropout": 0,
            "d_ff": 10240,
            "d_kv": 64,
            "d_model": 4096,
            "decoder_start_token_id": 0,
            "dense_act_fn": "gelu_new",
            "dropout_rate": 0.1,
            "eos_token_id": 1,
            "feed_forward_proj": "gated-gelu",
            "initializer_factor": 1.0,
            "is_encoder_decoder": True,
            "is_gated_act": True,
            "layer_norm_epsilon": 1e-6,
            "model_type": "t5",
            "num_decoder_layers": 24,
            "num_heads": 64,
            "num_layers": 24,
            "output_past": True,
            "pad_token_id": 0,
            "relative_attention_max_distance": 128,
            "relative_attention_num_buckets": 32,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 32128
        }

class T5XXLTextEncoderInt8(T5XXLTextEncoder):
    """
    T5-XXL Text Encoder model with 8-Bit quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-t5-xxl.int8.bf16.safetensors"
    quantization = "bitsandbytes_8bit"
    pre_quantized = True

class T5XXLTextEncoderNF4(T5XXLTextEncoder):
    """
    T5-XXL Text Encoder model with NF4 quantization.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-t5-xxl.nf4.bf16.safetensors"
    quantization = "bitsandbytes_nf4"
    pre_quantized = True
