from ..task import WhisperAudioTranscription
from ..pretrained import (
    WhisperV1Tokenizer,
    WhisperV1FeatureExtractor,
)
from .model import WhisperBaseModel

__all__ = ["WhisperBaseAudioTranscription"]

class WhisperBaseAudioTranscription(WhisperAudioTranscription):
    """
    Uses whisper to transcribe audio.
    """

    """Global Task Metadata"""
    task = "audio-transcription"
    model = "whisper-base"
    default = False
    display_name = "Whisper Base Audio Transcription"
    static_gpu_memory_gb = 292.60 / 1024 # 292.60 MB
    pretrained_models = {
        "model": WhisperBaseModel,
        "tokenizer": WhisperV1Tokenizer,
        "feature_extractor": WhisperV1FeatureExtractor,
    }
