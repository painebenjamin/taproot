from ..pretrained import DistilledWhisperEnglishTokenizer, DistilledWhisperEnglishFeatureExtractor
from ..task import DistilledWhisperAudioTranscription
from .model import DistilledWhisperSmallEnglishModel

__all__ = ["DistilledWhisperSmallEnglishAudioTranscription"]

class DistilledWhisperSmallEnglishAudioTranscription(DistilledWhisperAudioTranscription):
    """
    Uses whisper to transcribe audio.
    """

    """Global Task Metadata"""
    task = "audio-transcription"
    model = "distilled-whisper-small-english"
    default = False
    display_name = "Distilled Whisper Small (English) Audio Transcription"
    static_gpu_memory_gb = 664.59 / 1024 # 664.59 MB
    pretrained_models = {
        "model": DistilledWhisperSmallEnglishModel,
        "tokenizer": DistilledWhisperEnglishTokenizer,
        "feature_extractor": DistilledWhisperEnglishFeatureExtractor,
    }

    @property
    def use_generate_kwargs(self) -> bool:
        """
        Returns whether to use generate_kwargs for the model.
        """
        return False
