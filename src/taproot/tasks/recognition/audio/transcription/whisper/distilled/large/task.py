from ...pretrained import WhisperV3Tokenizer, WhisperV3FeatureExtractor
from ..task import DistilledWhisperAudioTranscription
from .model import DistilledWhisperLargeV3Model

__all__ = ["DistilledWhisperLargeV3AudioTranscription"]

class DistilledWhisperLargeV3AudioTranscription(DistilledWhisperAudioTranscription):
    """
    Uses whisper to transcribe audio.
    """

    """Global Task Metadata"""
    task = "audio-transcription"
    model = "distilled-whisper-large-v3"
    default = True
    display_name = "Distilled Whisper Large V3 Audio Transcription"
    static_gpu_memory_gb = 1.51
    pretrained_models = {
        "model": DistilledWhisperLargeV3Model,
        "tokenizer": WhisperV3Tokenizer,
        "feature_extractor": WhisperV3FeatureExtractor,
    }
