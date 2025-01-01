from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING

from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor # type: ignore[import-not-found,import-untyped,unused-ignore]

__all__ = ["DistilledWhisperEnglishFeatureExtractor"]

class DistilledWhisperEnglishFeatureExtractor(PretrainedModelMixin):
    """
    Feature extractor for the Whisper v3 models.
    """
    @classmethod
    def get_model_class(cls) -> Type[WhisperFeatureExtractor]:
        """
        Returns the model class.
        """
        from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor # type: ignore[import-not-found,import-untyped,unused-ignore]
        return WhisperFeatureExtractor # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "chunk_length": 30,
            "feature_size": 80,
            "hop_length": 160,
            "n_fft": 400,
            "n_samples": 480000,
            "nb_max_frames": 3000,
            "padding_side": "right",
            "padding_value": 0.0,
            "return_attention_mask": False,
            "sampling_rate": 16000
        }
