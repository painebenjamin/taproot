from __future__ import annotations

from typing import Any, Dict, Optional, Type, List, Union, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from .model import DACAutoencoder, SpeakerEmbeddingLDA, ZonosConfig, Zonos

__all__ = [
    "ZonosAutoencoderModel",
    "ZonosSpeakerEmbeddingModel",
    "ZonosTransformerModel",
    "ZonosHybridModel",
]

class ZonosAutoencoderModel(PretrainedModelMixin):
    """
    Zonos' Autoencoder (using Descript Audio Codec, DAC).
    """
    model_url: Optional[Union[str, List[str]]] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/audio-vocoder-descript-44khz.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[DACAutoencoder]:
        """
        Returns the model class.
        """
        from .model import DACAutoencoder
        return DACAutoencoder

class ZonosSpeakerEmbeddingModel(PretrainedModelMixin):
    """
    Zonos Speaker Embedding model.
    """
    model_url: Optional[Union[str, List[str]]] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/audio-diarisation-zonos-speaker-embedding.safetensors"

    @classmethod
    def get_model_class(cls) -> Type[SpeakerEmbeddingLDA]:
        """
        Returns the model class.
        """
        from .model import SpeakerEmbeddingLDA
        return SpeakerEmbeddingLDA

class ZonosHybridModel(PretrainedModelMixin):
    """
    Zonos hybrid model.
    """
    dtype = "bfloat16"
    model_url: Optional[Union[str, List[str]]] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/speech-synthesis-zonos-hybrid-v0-1.bf16.safetensors"

    @classmethod
    def get_config_class(cls) -> Type[ZonosConfig]:
        """
        Returns the configuration class.
        """
        from .model import ZonosConfig
        return ZonosConfig

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration.
        """
        from .model import BackboneConfig, PrefixConditionerConfig
        backbone_config = BackboneConfig(**{ # type: ignore[arg-type]
            "d_model": 2048,
            "d_intermediate": 0,
            "attn_mlp_d_intermediate": 8192,
            "n_layer": 46,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
            "attn_cfg": {
                "causal": True,
                "num_heads": 16,
                "num_heads_kv": 4,
                "rotary_emb_dim": 128,
                "qkv_proj_bias": False,
                "out_proj_bias": False
            },
            "rms_norm": False,
            "residual_in_fp32": False,
            "norm_epsilon": 1e-05
        })
        prefix_conditioner_config = PrefixConditionerConfig(**{ # type: ignore[arg-type]
            "conditioners": [
                {
                    "type": "EspeakPhonemeConditioner",
                    "name": "espeak"
                },
                {
                    "cond_dim": 128,
                    "uncond_type": "learned",
                    "projection": "linear",
                    "type": "PassthroughConditioner",
                    "name": "speaker"
                },
                {
                    "input_dim": 8,
                    "uncond_type": "learned",
                    "type": "FourierConditioner",
                    "name": "emotion"
                },
                {
                    "min_val": 0,
                    "max_val": 24000,
                    "uncond_type": "learned",
                    "type": "FourierConditioner",
                    "name": "fmax"
                },
                {
                    "min_val": 0,
                    "max_val": 400,
                    "uncond_type": "learned",
                    "type": "FourierConditioner",
                    "name": "pitch_std"
                },
                {
                    "min_val": 0,
                    "max_val": 40,
                    "uncond_type": "learned",
                    "type": "FourierConditioner",
                    "name": "speaking_rate"
                },
                {
                    "min_val": -1,
                    "max_val": 126,
                    "uncond_type": "learned",
                    "type": "IntegerConditioner",
                    "name": "language_id"
                },
                {
                    "input_dim": 8,
                    "min_val": 0.5,
                    "max_val": 0.8,
                    "uncond_type": "learned",
                    "type": "FourierConditioner",
                    "name": "vqscore_8"
                },
                {
                    "min_val": -1.0,
                    "max_val": 1000,
                    "uncond_type": "learned",
                    "type": "FourierConditioner",
                    "name": "ctc_loss"
                },
                {
                    "min_val": 1,
                    "max_val": 5,
                    "uncond_type": "learned",
                    "type": "FourierConditioner",
                    "name": "dnsmos_ovrl"
                },
                {
                    "min_val": 0,
                    "max_val": 1,
                    "uncond_type": "learned",
                    "type": "IntegerConditioner",
                    "name": "speaker_noised"
                }
            ],
            "projection": "linear"
        })

        return {
            "backbone": backbone_config,
            "prefix_conditioner": prefix_conditioner_config,
            "eos_token_id": 1024,
            "masked_token_id": 1025,
            "n_codebooks": 9,
        }

    @classmethod
    def get_model_class(cls) -> Type[Zonos]:
        """
        Returns the model class.
        """
        from .model import Zonos
        return Zonos

class ZonosTransformerModel(PretrainedModelMixin):
    """
    Zonos transformer model.
    """
    dtype = "bfloat16"
    model_url: Optional[Union[str, List[str]]] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/speech-synthesis-zonos-transformer-v0-1.bf16.safetensors"

    @classmethod
    def get_config_class(cls) -> Type[ZonosConfig]:
        """
        Returns the configuration class.
        """
        from .model import ZonosConfig
        return ZonosConfig

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration.
        """
        from .model import BackboneConfig, PrefixConditionerConfig
        backbone_config = BackboneConfig(**{ # type: ignore[arg-type]
            "d_model": 2048,
            "d_intermediate": 0,
            "attn_mlp_d_intermediate": 8192,
            "n_layer": 26,
            "ssm_cfg": {},
            "attn_layer_idx": list(range(26)),
            "attn_cfg": {
                "causal": True,
                "num_heads": 16,
                "num_heads_kv": 4,
                "rotary_emb_dim": 128,
                "rotary_emb_interleaved": True,
                "qkv_proj_bias": False,
                "out_proj_bias": False
            },
            "rms_norm": False,
            "residual_in_fp32": False,
            "norm_epsilon": 1e-05
        })
        prefix_conditioner_config = PrefixConditionerConfig(**{ # type: ignore[arg-type]
            "conditioners": [
                {
                    "type": "EspeakPhonemeConditioner",
                    "name": "espeak"
                },
                {
                    "cond_dim": 128,
                    "uncond_type": "learned",
                    "projection": "linear",
                    "type": "PassthroughConditioner",
                    "name": "speaker"
                },
                {
                    "input_dim": 8,
                    "uncond_type": "learned",
                    "type": "FourierConditioner",
                    "name": "emotion"
                },
                {
                    "min_val": 0,
                    "max_val": 24000,
                    "uncond_type": "learned",
                    "type": "FourierConditioner",
                    "name": "fmax"
                },
                {
                    "min_val": 0,
                    "max_val": 400,
                    "uncond_type": "learned",
                    "type": "FourierConditioner",
                    "name": "pitch_std"
                },
                {
                    "min_val": 0,
                    "max_val": 40,
                    "uncond_type": "learned",
                    "type": "FourierConditioner",
                    "name": "speaking_rate"
                },
                {
                    "min_val": -1,
                    "max_val": 126,
                    "uncond_type": "learned",
                    "type": "IntegerConditioner",
                    "name": "language_id"
                },
            ],
            "projection": "linear"
        })

        return {
            "backbone": backbone_config,
            "prefix_conditioner": prefix_conditioner_config,
            "eos_token_id": 1024,
            "masked_token_id": 1025,
            "n_codebooks": 9,
        }

    @classmethod
    def get_model_class(cls) -> Type[Zonos]:
        """
        Returns the model class.
        """
        from .model import Zonos
        return Zonos
