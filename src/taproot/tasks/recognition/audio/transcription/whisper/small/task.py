from ..task import WhisperAudioTranscription
from ..pretrained import (
    WhisperV1Tokenizer,
    WhisperV1FeatureExtractor,
)
from .model import WhisperSmallModel

__all__ = ["WhisperSmallAudioTranscription"]

class WhisperSmallAudioTranscription(WhisperAudioTranscription):
    """
    Uses whisper to transcribe audio.
    """

    """Global Task Metadata"""
    task = "audio-transcription"
    model = "whisper-small"
    default = False
    display_name = "Whisper Small Audio Transcription"
    static_gpu_memory_gb = 967.71 / 1024 # 967.71 MB
    pretrained_models = {
        "model": WhisperSmallModel,
        "tokenizer": WhisperV1Tokenizer,
        "feature_extractor": WhisperV1FeatureExtractor,
    }
