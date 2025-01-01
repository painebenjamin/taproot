from ..task import WhisperAudioTranscription
from ..pretrained import (
    WhisperV1Tokenizer,
    WhisperV1FeatureExtractor,
)
from .model import WhisperTinyModel

__all__ = ["WhisperTinyAudioTranscription"]

class WhisperTinyAudioTranscription(WhisperAudioTranscription):
    """
    Uses whisper to transcribe audio.
    """

    """Global Task Metadata"""
    task = "audio-transcription"
    model = "whisper-tiny"
    default = False
    display_name = "Whisper Tiny Audio Transcription"
    static_gpu_memory_gb = 151.4 / 1024 # 151.4 MB
    pretrained_models = {
        "model": WhisperTinyModel,
        "tokenizer": WhisperV1Tokenizer,
        "feature_extractor": WhisperV1FeatureExtractor,
    }
