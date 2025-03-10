from typing import Any, Dict, Optional

from ...pretrained import WhisperModel

__all__ = ["DistilledWhisperMediumEnglishModel"]

class DistilledWhisperMediumEnglishModel(WhisperModel):
    """
    Inference model for the Distilled Whisper Medium English model.
    """

    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/audio-transcription-distilled-whisper-medium-english.safetensors"

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "activation_dropout": 0,
            "activation_function": "gelu",
            "apply_spec_augment": False,
            "architectures": ["WhisperForConditionalGeneration"],
            "attention_dropout": 0,
            "begin_suppress_tokens": [220, 50256],
            "bos_token_id": 50257,
            "classifier_proj_size": 256,
            "d_model": 1024,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": 4096,
            "decoder_layerdrop": 0,
            "decoder_layers": 2,
            "decoder_start_token_id": 50257,
            "dropout": 0,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "encoder_layerdrop": 0,
            "encoder_layers": 24,
            "eos_token_id": 50256,
            "forced_decoder_ids": [[1, 50362]],
            "init_std": 0.02,
            "is_encoder_decoder": True,
            "mask_feature_length": 10,
            "mask_feature_min_masks": 0,
            "mask_feature_prob": 0,
            "mask_time_length": 10,
            "mask_time_min_masks": 2,
            "mask_time_prob": 0.05,
            "max_length": 448,
            "max_source_positions": 1500,
            "max_target_positions": 448,
            "median_filter_width": 7,
            "model_type": "whisper",
            "num_hidden_layers": 24,
            "num_mel_bins": 80,
            "pad_token_id": 50256,
            "scale_embedding": False,
            "suppress_tokens": [
                1,
                2,
                7,
                8,
                9,
                10,
                14,
                25,
                26,
                27,
                28,
                29,
                31,
                58,
                59,
                60,
                61,
                62,
                63,
                90,
                91,
                92,
                93,
                357,
                366,
                438,
                532,
                685,
                705,
                796,
                930,
                1058,
                1220,
                1267,
                1279,
                1303,
                1343,
                1377,
                1391,
                1635,
                1782,
                1875,
                2162,
                2361,
                2488,
                3467,
                4008,
                4211,
                4600,
                4808,
                5299,
                5855,
                6329,
                7203,
                9609,
                9959,
                10563,
                10786,
                11420,
                11709,
                11907,
                13163,
                13697,
                13700,
                14808,
                15306,
                16410,
                16791,
                17992,
                19203,
                19510,
                20724,
                22305,
                22935,
                27007,
                30109,
                30420,
                33409,
                34949,
                40283,
                40493,
                40549,
                47282,
                49146,
                50257,
                50357,
                50358,
                50359,
                50360,
                50361,
            ],
            "torch_dtype": "float32",
            "use_cache": True,
            "use_weighted_layer_sum": False,
            "vocab_size": 51864,
        }

    @classmethod
    def get_generation_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default generation configuration for the model.
        """
        return {
            "begin_suppress_tokens": [220, 50256],
            "bos_token_id": 50257,
            "decoder_start_token_id": 50257,
            "eos_token_id": 50256,
            "is_multilingual": False,
            "max_initial_timestamp_index": 50,
            "max_length": 448,
            "no_timestamps_token_id": 50362,
            "pad_token_id": 50256,
            "prev_sot_token_id": 50360,
            "return_timestamps": False,
            "suppress_tokens": [
                1,
                2,
                7,
                8,
                9,
                10,
                14,
                25,
                26,
                27,
                28,
                29,
                31,
                58,
                59,
                60,
                61,
                62,
                63,
                90,
                91,
                92,
                93,
                357,
                366,
                438,
                532,
                685,
                705,
                796,
                930,
                1058,
                1220,
                1267,
                1279,
                1303,
                1343,
                1377,
                1391,
                1635,
                1782,
                1875,
                2162,
                2361,
                2488,
                3467,
                4008,
                4211,
                4600,
                4808,
                5299,
                5855,
                6329,
                7203,
                9609,
                9959,
                10563,
                10786,
                11420,
                11709,
                11907,
                13163,
                13697,
                13700,
                14808,
                15306,
                16410,
                16791,
                17992,
                19203,
                19510,
                20724,
                22305,
                22935,
                27007,
                30109,
                30420,
                33409,
                34949,
                40283,
                40493,
                40549,
                47282,
                49146,
                50257,
                50357,
                50358,
                50359,
                50360,
                50361,
            ],
            "use_scan": False,
        }
