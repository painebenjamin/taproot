from __future__ import annotations

from taproot.util import PretrainedModelMixin

from typing import Optional, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import ( # type: ignore[import-untyped]
        CLIPVisionConfig,
        CLIPVisionModelWithProjection
    )

__all__ = [
    "OpenCLIPViTHVisionEncoder",
    "OpenCLIPViTBigGVisionEncoder"
]

class OpenCLIPViTHVisionEncoder(PretrainedModelMixin):
    """
    OpenCLIP ViT-H Vision Encoder.
    """
    model_url = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors?filename=vision-encoding-clip-vit-h.fp16.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[CLIPVisionModelWithProjection]:
        """
        Get the model class for the Stable Diffusion XL Text Encoder.
        """
        from transformers import CLIPVisionModelWithProjection
        return CLIPVisionModelWithProjection # type: ignore[no-any-return]

    @classmethod
    def get_config_class(cls) -> Type[CLIPVisionConfig]:
        """
        Get the configuration class for the Stable Diffusion XL Vision Encoder.
        """
        from transformers import CLIPVisionConfig
        return CLIPVisionConfig # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Stable Diffusion XL Text Encoder.
        """
        return {
            "attention_dropout": 0,
            "dropout": 0,
            "hidden_act": "gelu",
            "hidden_size": 1280,
            "image_size": 224,
            "initializer_factor": 1,
            "initializer_range": 0.02,
            "intermediate_size": 5120,
            "layer_norm_eps": 0.00001,
            "model_type": "clip_vision_model",
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 32,
            "patch_size": 14,
            "projection_dim": 1024,
            "torch_dtype": "float16",
        }


class OpenCLIPViTBigGVisionEncoder(PretrainedModelMixin):
    """
    OpenCLIP ViT-BigG Vision Encoder.
    """
    model_url = "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors?filename=vision-encoding-clip-vit-big-g.fp16.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[CLIPVisionModelWithProjection]:
        """
        Get the model class for the Stable Diffusion XL Text Encoder.
        """
        from transformers import CLIPVisionModelWithProjection
        return CLIPVisionModelWithProjection # type: ignore[no-any-return]

    @classmethod
    def get_config_class(cls) -> Type[CLIPVisionConfig]:
        """
        Get the configuration class for the Stable Diffusion XL Vision Encoder.
        """
        from transformers import CLIPVisionConfig
        return CLIPVisionConfig # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Stable Diffusion XL Text Encoder.
        """
        return {
            "add_cross_attention": False,
            "attention_dropout": 0,
            "bad_words_ids": None,
            "begin_suppress_tokens": None,
            "bos_token_id": None,
            "chunk_size_feed_forward": 0,
            "cross_attention_hidden_size": None,
            "decoder_start_token_id": None,
            "diversity_penalty": 0,
            "do_sample": False,
            "dropout": 0,
            "early_stopping": False,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": None,
            "exponential_decay_length_penalty": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "hidden_act": "gelu",
            "hidden_size": 1664,
            "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1"
            },
            "image_size": 224,
            "initializer_factor": 1,
            "initializer_range": 0.02,
            "intermediate_size": 8192,
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1
            },
            "layer_norm_eps": 0.00001,
            "length_penalty": 1,
            "max_length": 20,
            "min_length": 0,
            "model_type": "clip_vision_model",
            "no_repeat_ngram_size": 0,
            "num_attention_heads": 16,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_channels": 3,
            "num_hidden_layers": 48,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": None,
            "patch_size": 14,
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1,
            "return_dict": True,
            "return_dict_in_generate": False,
            "sep_token_id": None,
            "suppress_tokens": None,
            "task_specific_params": None,
            "temperature": 1,
            "tf_legacy_loss": False,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": True,
            "tokenizer_class": None,
            "top_k": 50,
            "top_p": 1,
            "torch_dtype": None,
            "torchscript": False,
            "typical_p": 1,
            "use_bfloat16": False,
            "projection_dim": 1280
        }
