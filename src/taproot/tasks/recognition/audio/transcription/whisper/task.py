from __future__ import annotations

from typing import Optional, Dict, Literal, Any, Union, List, TYPE_CHECKING

from taproot.constants import *
from taproot.tasks.base import Task
from taproot.util import audio_to_bct_tensor, is_multi_audio

if TYPE_CHECKING:
    import torch
    from taproot.hinting import AudioType
    from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline # type: ignore[import-untyped,import-not-found,unused-ignore]
    from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration # type: ignore[import-untyped,import-not-found,unused-ignore]
    from transformers.models.whisper.tokenization_whisper import WhisperTokenizer # type: ignore[import-untyped,import-not-found,unused-ignore]
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor # type: ignore[import-untyped,import-not-found,unused-ignore]

__all__ = ["WhisperAudioTranscription"]

class WhisperAudioTranscription(Task):
    """
    Uses whisper to transcribe audio.
    This parent class should be inherited by specific models.
    """

    """Authorship metadata"""
    author = "Alec Radford"
    author_additional = ["Jong Wook Kim", "Tao Xu", "Greg Brockman", "Christine McLeavey", "Ilya Sutskever"]
    author_url = "https://arxiv.org/abs/2212.04356"
    author_affiliations = ["OpenAI"]
    author_journal = "arXiv"
    author_journal_volume = "2212.04356"
    author_journal_uear = 2022
    author_journal_title = "Robust Speech Recognition via Large-Scale Weak Supervision"

    """License metadata"""
    license = LICENSE_APACHE

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        The required packages for the task.
        """
        return {
            "accelerate": ACCELERATE_VERSION_SPEC,
            "transformers": TRANSFORMERS_VERSION_SPEC,
            "torch": TORCH_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
            "einops": EINOPS_VERSION_SPEC,
            "tokenizers": TOKENIZERS_VERSION_SPEC,
            "torchaudio": TORCHAUDIO_VERSION_SPEC,
            "torchvision": TORCHVISION_VERSION_SPEC,
            "scipy": SCIPY_VERSION_SPEC,
            "safetensors": SAFETENSORS_VERSION_SPEC,
            "pil": PILLOW_VERSION_SPEC,
            "sklearn": SKLEARN_VERSION_SPEC,
        }

    """Internal Task Attributes"""

    @property
    def whisper(self) -> WhisperForConditionalGeneration:
        """
        Add mapped modules as properties for convenience and type hinting.
        """
        return self.pretrained.model

    @property
    def tokenizer(self) -> WhisperTokenizer:
        """
        Add mapped modules as properties for convenience and type hinting.
        """
        return self.pretrained.tokenizer

    @property
    def feature_extractor(self) -> WhisperFeatureExtractor:
        """
        Add mapped modules as properties for convenience and type hinting.
        """
        return self.pretrained.feature_extractor

    @property
    def pipeline(self) -> AutomaticSpeechRecognitionPipeline:
        """
        Get the pipeline for the task.
        """
        if not hasattr(self, "_pipeline"):
            from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline # type: ignore[import-untyped,import-not-found,unused-ignore]
            self._pipeline = AutomaticSpeechRecognitionPipeline(
                model=self.whisper,
                feature_extractor=self.feature_extractor,
                tokenizer=self.tokenizer,
                torch_dtype=self.dtype,
            )
        return self._pipeline

    @property
    def use_generate_kwargs(self) -> bool:
        """
        Returns whether to use generate_kwargs for the model.
        """
        return True

    def resample(self, audio: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """
        Resample the audio tensor to the model's sampling rate.
        """
        import torch
        import torchaudio # type: ignore[import-untyped,import-not-found,unused-ignore]
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=self.feature_extractor.sampling_rate)
        return torch.stack([resampler(audio_datum) for audio_datum in audio])

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        audio: AudioType,
        sample_rate: Optional[int]=None,
        timestamps: Optional[Literal["chunk", "word"]]=None,
        max_new_tokens: Optional[int]=None,
        chunk_length_s: float=30.0,
        batch_size: int=32,
    ) -> Union[str, List[str]]:
        """
        Transcribes audio to text.

        :param audio: The audio to transcribe.
        :param sample_rate: The sample rate of the audio.
        :param timestamps: How to timestamp the output, if at all.
        :param max_new_tokens: The maximum number of new tokens to generate.
        :param chunk_length_s: The length of the audio chunks to process. Longer chunks may be faster but less accurate.
        :param batch_size: The batch size to use for processing the audio.
        :returns: The transcribed text.
        """
        audio_data, sampling_rate = audio_to_bct_tensor(
            audio,
            self.feature_extractor.sampling_rate if sample_rate is None else sample_rate,
        )

        if isinstance(sampling_rate, int) and sampling_rate != self.feature_extractor.sampling_rate:
            audio_data = self.resample(audio_data, sampling_rate)

        kwargs: Dict[str, Any] = {
            "chunk_length_s": chunk_length_s,
            "batch_size": batch_size,
            "return_timestamps": timestamps,
            "max_new_tokens": max_new_tokens,
        }
        if self.use_generate_kwargs:
            kwargs["generate_kwargs"] = {
                "task": "transcribe",
                "language": None
            }
        outputs = [
            self.pipeline(audio_datum.mean(0).numpy(), **kwargs)["text"].strip()
            for audio_datum in audio_data
        ]
        if is_multi_audio(audio):
            return outputs
        return outputs[0] # type: ignore[no-any-return]
