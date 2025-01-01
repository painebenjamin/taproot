from __future__ import annotations

from taproot.util import PretrainedModelMixin

from typing import Optional, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import CLIPConfig # type: ignore[import-untyped,import-not-found,unused-ignore]
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker as SafetyCheckerPipeline

__all__ = [
    "StableDiffusionSafetyChecker"
]

class StableDiffusionSafetyChecker(PretrainedModelMixin):
    """
    The model for the Stable Diffusion SafetyChecker.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-generation-stable-diffusion-v1-5-safety-checker.fp16.safetensors"
    use_strict = False

    @classmethod
    def get_config_class(cls) -> Type[CLIPConfig]:
        """
        Gets the configuration class for the Stable Diffusion SafetyChecker.
        """
        from transformers import CLIPConfig # type: ignore[import-untyped,import-not-found,unused-ignore]
        return CLIPConfig # type: ignore[no-any-return]

    @classmethod
    def get_model_class(cls) -> Type[SafetyCheckerPipeline]:
        """
        Get the model class for the Stable Diffusion Safety Checker.
        """
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker as SafetyCheckerPipeline
        return SafetyCheckerPipeline

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Stable Diffusion SafetyChecker.
        """
        return {
            "initializer_factor": 1,
            "logit_scale_init_value": 2.6592,
            "model_type": "clip",
            "projection_dim": 768,
            "text_config": {
                "add_cross_attention": False,
                "architectures": None,
                "attention_dropout": 0,
                "bad_words_ids": None,
                "bos_token_id": 0,
                "chunk_size_feed_forward": 0,
                "cross_attention_hidden_size": None,
                "decoder_start_token_id": None,
                "diversity_penalty": 0,
                "do_sample": False,
                "dropout": 0,
                "early_stopping": False,
                "encoder_no_repeat_ngram_size": 0,
                "eos_token_id": 2,
                "exponential_decay_length_penalty": None,
                "finetuning_task": None,
                "forced_bos_token_id": None,
                "forced_eos_token_id": None,
                "hidden_act": "quick_gelu",
                "hidden_size": 768,
                "id2label": {
                    "0": "LABEL_0",
                    "1": "LABEL_1"
                },
                "initializer_factor": 1,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "is_decoder": False,
                "is_encoder_decoder": False,
                "label2id": {
                    "LABEL_0": 0,
                    "LABEL_1": 1
                },
                "layer_norm_eps": 0.00001,
                "length_penalty": 1,
                "max_length": 20,
                "max_position_embeddings": 77,
                "min_length": 0,
                "model_type": "clip_text_model",
                "no_repeat_ngram_size": 0,
                "num_attention_heads": 12,
                "num_beam_groups": 1,
                "num_beams": 1,
                "num_hidden_layers": 12,
                "num_return_sequences": 1,
                "output_attentions": False,
                "output_hidden_states": False,
                "output_scores": False,
                "pad_token_id": 1,
                "prefix": None,
                "problem_type": None,
                "pruned_heads": {},
                "remove_invalid_values": False,
                "repetition_penalty": 1,
                "return_dict": True,
                "return_dict_in_generate": False,
                "sep_token_id": None,
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
                "vocab_size": 49408
            },
            "text_config_dict": {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "num_hidden_layers": 12
            },
            "torch_dtype": "float32",
            "transformers_version": None,
            "vision_config": {
                "add_cross_attention": False,
                "architectures": None,
                "attention_dropout": 0,
                "bad_words_ids": None,
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
                "hidden_act": "quick_gelu",
                "hidden_size": 1024,
                "id2label": {
                    "0": "LABEL_0",
                    "1": "LABEL_1"
                },
                "image_size": 224,
                "initializer_factor": 1,
                "initializer_range": 0.02,
                "intermediate_size": 4096,
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
                "num_hidden_layers": 24,
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
                "use_bfloat16": False
            },
            "vision_config_dict": {
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "patch_size": 14
            }
        }
