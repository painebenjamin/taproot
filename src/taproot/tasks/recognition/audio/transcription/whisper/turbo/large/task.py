from ...task import WhisperAudioTranscription
from ...pretrained import WhisperV3Tokenizer, WhisperV3FeatureExtractor
from .model import TurboWhisperLargeV3Model

__all__ = ["TurboWhisperLargeV3AudioTranscription"]

class TurboWhisperLargeV3AudioTranscription(WhisperAudioTranscription):
    """
    Uses whisper to transcribe audio.
    """

    """Global Task Metadata"""
    task = "audio-transcription"
    model = "turbo-whisper-large-v3"
    default = False
    display_name = "Turbo Whisper Large V3 Audio Transcription"
    static_memory_gb = 0.09614 # 96.14 MB
    static_gpu_memory_gb = 1.62
    pretrained_models = {
        "model": TurboWhisperLargeV3Model,
        "tokenizer": WhisperV3Tokenizer,
        "feature_extractor": WhisperV3FeatureExtractor,
    }
