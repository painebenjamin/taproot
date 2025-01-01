from ..task import WhisperAudioTranscription
from ..pretrained import (
    WhisperV1Tokenizer,
    WhisperV1FeatureExtractor,
)

from .model import WhisperMediumModel

__all__ = ["WhisperMediumAudioTranscription"]

class WhisperMediumAudioTranscription(WhisperAudioTranscription):
    """
    Uses whisper to transcribe audio.
    """

    """Global Task Metadata"""
    task = "audio-transcription"
    model = "whisper-medium"
    default = False
    display_name = "Whisper Medium Audio Transcription"
    static_gpu_memory_gb = 3.06
    pretrained_models = {
        "model": WhisperMediumModel,
        "tokenizer": WhisperV1Tokenizer,
        "feature_extractor": WhisperV1FeatureExtractor,
    }
