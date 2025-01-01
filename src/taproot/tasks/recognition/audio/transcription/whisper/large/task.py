from ..task import WhisperAudioTranscription
from ..pretrained import (
    WhisperV3Tokenizer,
    WhisperV3FeatureExtractor,
)
from .model import WhisperLargeV3Model

__all__ = ["WhisperLargeV3AudioTranscription"]

class WhisperLargeV3AudioTranscription(WhisperAudioTranscription):
    """
    Uses whisper to transcribe audio.
    """

    """Global Task Metadata"""
    task = "audio-transcription"
    model = "whisper-large-v3"
    display_name = "Whisper Large V3 Audio Transcription"
    static_gpu_memory_gb = 3.09
    pretrained_models = {
        "model": WhisperLargeV3Model,
        "tokenizer": WhisperV3Tokenizer,
        "feature_extractor": WhisperV3FeatureExtractor,
    }
