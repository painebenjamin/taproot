from ..task import WhisperAudioTranscription

__all__ = ["DistilledWhisperAudioTranscription"]

class DistilledWhisperAudioTranscription(WhisperAudioTranscription):
    """
    Updated authorship information
    """
    author = "Sanchit Gandhi"
    author_url = "https://arxiv.org/abs/2311.00430"
    author_affiliations = ["Hugging Face"]
    author_additional = ["Patrick von Platen", "Alexander M. Rush"]
    author_journal = "arXiv"
    author_journal_volume = "2311.00430"
    author_journal_year = 2023
    author_journal_title = "Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling"
