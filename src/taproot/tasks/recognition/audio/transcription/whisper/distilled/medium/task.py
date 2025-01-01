from ..pretrained import DistilledWhisperEnglishFeatureExtractor, DistilledWhisperEnglishTokenizer
from ..task import DistilledWhisperAudioTranscription
from .model import DistilledWhisperMediumEnglishModel

__all__ = ["DistilledWhisperMediumEnglishAudioTranscription"]

class DistilledWhisperMediumEnglishAudioTranscription(DistilledWhisperAudioTranscription):
    """
    Uses whisper to transcribe audio.
    """

    """Global Task Metadata"""
    task = "audio-transcription"
    model = "distilled-whisper-medium-english"
    default = False
    display_name = "Distilled Whisper Medium (English) Audio Transcription"
    static_gpu_memory_gb = 1.58
    pretrained_models = {
        "model": DistilledWhisperMediumEnglishModel,
        "tokenizer": DistilledWhisperEnglishTokenizer,
        "feature_extractor": DistilledWhisperEnglishFeatureExtractor,
    }

    @property
    def use_generate_kwargs(self) -> bool:
        """
        Returns whether to use generate_kwargs for the model.
        """
        return False
