from __future__ import annotations

import re

from typing import Optional, Union, List, Dict, Any, Tuple, TYPE_CHECKING

from taproot.constants import *
from taproot.util import (
    logger,
    audio_to_bct_tensor,
    concatenate_audio,
    seed_everything,
    simplify_quotations
)

from taproot.tasks.base import Task

# Pretrained models
from .pretrained import (
    F5TTSModel,
    F5TTSVocoder
)

# Components (sub-tasks)
from taproot.tasks.transformation import DeepFilterNet3Enhancement
from taproot.tasks.recognition import DistilledWhisperLargeV3AudioTranscription

if TYPE_CHECKING:
    import torch
    from vocos import Vocos # type: ignore[import-not-found,import-untyped,unused-ignore]
    from taproot.modeling import ConditionalFlowMatching
    from taproot.hinting import AudioType, AudioResultType, SeedType

__all__ = ["F5TTSSpeechSynthesis"]

class F5TTSSpeechSynthesis(Task):
    """
    Speech synthesis task using the F5TTS model.
    """

    """Global Task Metadata"""

    task: str = "speech-synthesis"
    model: Optional[str] = "f5tts"
    default: bool = False
    display_name = "F5TTS Speech Synthesis"
    pretrained_models = {
        "model": F5TTSModel,
        "vocoder": F5TTSVocoder
    }
    optional_component_tasks = {
        "enhance": DeepFilterNet3Enhancement,
        "transcribe": DistilledWhisperLargeV3AudioTranscription
    }
    static_memory_gb = 0.10766 # 107.66 MB
    static_gpu_memory_gb = 0.70716 # 707.16 MB

    """Authorship Metadata"""
    author = "Yushen Chen"
    author_url = "https://arxiv.org/abs/2410.06885"
    author_additional = ["Zhikang Niu", "Ziyang Ma", "Keqi Deng", "Chunhui Wang", "Jian Zhao", "Kai Yu", "Xie Chen"]
    author_journal = "arXiv"
    author_journal_volume = "2410.06885"
    author_journal_year = 2024
    author_journal_title = "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching"

    """License Metadata"""
    license = LICENSE_CC_BY_NC_4

    """Task-Specific Metadata"""
    sample_rate: int = 24000
    hop_length: int = 256

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Required packages.
        """
        return {
            "accelerate": ACCELERATE_VERSION_SPEC,
            "torchvision": TORCHVISION_VERSION_SPEC,
            "pil": PILLOW_VERSION_SPEC,
            "torch": TORCH_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
            "torchaudio": TORCHAUDIO_VERSION_SPEC,
            "librosa": LIBROSA_VERSION_SPEC,
            "einops": EINOPS_VERSION_SPEC,
            "scipy": SCIPY_VERSION_SPEC,
            "safetensors": SAFETENSORS_VERSION_SPEC,
            "transformers": TRANSFORMERS_VERSION_SPEC,
            "sklearn": SKLEARN_VERSION_SPEC,
            "torchdiffeq": "~=0.2",
            "vocos": "~=0.1",
        }

    """Internal Task Attributes"""

    @property
    def f5tts(self) -> ConditionalFlowMatching:
        """
        Add mapped modules as properties for convenience and type hinting.
        """
        return self.pretrained.model # type: ignore[no-any-return]

    @property
    def vocoder(self) -> Vocos:
        """
        Add mapped modules as properties for convenience and type hinting.
        """
        return self.pretrained.vocoder

    """Override Properties"""

    @property
    def last_intermediate(self) -> Any:
        """
        Override last intermediates to concatenate instead of return the last.
        """
        if self.intermediates:
            # No cross-fading for intermediates
            return concatenate_audio(self.intermediates)
        return None

    """Public Task Methods"""

    def get_default_reference(self) -> Tuple[torch.Tensor, str, int]:
        """
        Get the default reference audio, text, and sample rate.
        """
        return (
            torch.zeros(1, 1, 1),
            "This is a reference audio.",
            self.sample_rate
        )

    def get_sample_rate(self, enhance: bool=False) -> int:
        """
        Get the sample rate of the model.
        """
        if enhance:
            return self.tasks.enhance.df_state.sr() # type: ignore[no-any-return]
        return self.sample_rate

    def ends_with_multi_byte_character(self, text: str) -> bool:
        """
        Check if the text ends with a multi-byte character.
        """
        return len(text[-1].encode("utf-8")) > 1

    def get_punctuation_pause(self, text: str) -> float:
        """
        Check if the text ends with punctuation and return the pause duration ratio.
        """
        char = text.strip()[-1]
        if char in [".", "!", "?", "。", "，", "！", "？"]: 
            return 1.0
        if char in [",", ";", "；"]:
            return 0.5
        return 0.0

    def format_text(
        self,
        text: str,
        is_reference_text: bool=False
    ) -> str:
        """
        Format text for synthesis.

        :param text: The text to format.
        :param is_reference_text: Whether the text is a reference text. Default is False.
        :return: The formatted text.
        """
        text = simplify_quotations(text).replace(";", ",")
        if is_reference_text:
            text = text.strip(",;- ")
            if not text.endswith("."):
                return f"{text}. "
            return f"{text} "
        return text

    def chunk_text(
        self,
        text: str,
        max_length: int=135
    ) -> List[str]:
        """
        Chunk text into smaller pieces.
        """
        chunks = []

        current_chunk = ""
        current_chunk_length = 0

        # Split the text into sentences based on punctuation followed by whitespace
        sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

        for sentence in sentences:
            encoded = sentence.encode("utf-8")
            encoded_length = len(encoded)

            if encoded_length == 0:
                continue

            if current_chunk_length + encoded_length <= max_length:
                current_chunk += sentence
                current_chunk_length += encoded_length
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_chunk_length = encoded_length

            if not self.ends_with_multi_byte_character(sentence):
                current_chunk += " "
                current_chunk_length += 1

        if current_chunk_length > 0:
            chunks.append(current_chunk.strip())

        return chunks

    def enhance(
        self,
        audio: torch.Tensor,
        seed: SeedType=None,
    ) -> torch.Tensor:
        """
        Enhance audio using the DeepFilterNet3 model.
        """
        return self.tasks.enhance( # type: ignore[no-any-return]
            audio=audio,
            sample_rate=self.sample_rate,
            seed=seed,
            output_format="float"
        )[0]

    def transcribe(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int]=None
    ) -> str:
        """
        Transcribe audio using the Whisper model.
        """
        return self.tasks.transcribe( # type: ignore[no-any-return]
            audio=audio,
            sample_rate=self.sample_rate if sample_rate is None else sample_rate,
        )[0]

    def synthesize(
        self,
        texts: List[str],
        reference_text: str,
        reference_audio: torch.Tensor,
        reference_sample_rate: int,
        enhance: bool=False,
        seed: SeedType=None,
        speed: float=1.0,
        sway_sampling_coef: float=-1.0,
        target_rms: float=0.1,
        cross_fade_duration: float=0.15,
        punctuation_pause_duration: float=0.0,
        num_steps: int=32,
        cfg_strength: float=2.0,
        fix_duration: Optional[float]=None,
        stream: bool=False,
    ) -> torch.Tensor:
        """
        Synthesize audio from text.
        """
        import torch
        import torchaudio # type: ignore[import-untyped]

        rms = torch.sqrt(torch.mean(torch.square(reference_audio)))
        if rms < target_rms:
            reference_audio = reference_audio * (target_rms / rms)

        if reference_sample_rate != self.sample_rate:
            reference_audio = torchaudio.transforms.Resample(
                orig_freq=reference_sample_rate,
                new_freq=self.sample_rate
            )(reference_audio)

        reference_audio = reference_audio.to(self.f5tts.device)
        reference_text = self.format_text(reference_text, is_reference_text=True)
        reference_text_length = len(reference_text.encode("utf-8"))

        if reference_audio.ndim == 3:
            # Remove batch dimension
            reference_audio = reference_audio.squeeze(0)

        audios: List[torch.Tensor] = []
        num_texts = len(texts)

        for i, text in enumerate(texts):
            text = self.format_text(text)

            text_chunks = self.chunk_text(text)
            num_text_chunks = len(text_chunks)

            for j, text_chunk in enumerate(text_chunks):
                reference_audio_length = reference_audio.shape[-1] // self.hop_length

                if fix_duration is not None:
                    duration = int(fix_duration * self.sample_rate / self.hop_length)
                else:
                    # Estimate duration
                    reference_text_length = len(reference_text.encode("utf-8"))
                    generate_text_length = len(text_chunk.encode("utf-8"))
                    duration = reference_audio_length + int(reference_audio_length / reference_text_length * generate_text_length / speed)

                logger.debug(f"Generating audio from text chunk {j + 1} of {num_text_chunks} for text {i + 1} of {num_texts} with duration {duration}. Reference text: {reference_text}, text chunk: {text_chunk}")
                audio, trajectory = self.f5tts.sample(
                    cond=reference_audio,
                    text=[reference_text + text_chunk],
                    duration=duration,
                    steps=num_steps,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef
                )

                vocoder_dtype = next(self.vocoder.parameters()).dtype
                audio = audio.to(dtype=vocoder_dtype)[:, reference_audio_length:, :]
                audio = audio.permute(0, 2, 1)
                audio = self.vocoder.decode(audio).to(dtype=torch.float32)

                if rms < target_rms:
                    audio = audio * rms / target_rms

                audio = audio.squeeze().cpu()

                if enhance:
                    audio = self.enhance(audio, seed=seed)

                if (i < num_texts - 1 or j < num_text_chunks - 1):
                    pause = self.get_punctuation_pause(text_chunk)
                    if pause > 0:
                        pause_duration = punctuation_pause_duration * pause
                        logger.debug(f"Adding pause of {pause_duration} seconds to the end of the audio.")
                        num_pause_samples = int(pause_duration * self.get_sample_rate(enhance=enhance) * (1 + (enhance * 7)))
                        pause_samples = torch.zeros(num_pause_samples).to(dtype=audio.dtype, device=audio.device)
                        audio = torch.cat([audio, pause_samples])

                if stream:
                    self.add_intermediate(audio)

                audios.append(audio)

        # Combine audios
        return concatenate_audio(
            audios,
            cross_fade_duration=cross_fade_duration,
            sample_rate=self.get_sample_rate(enhance=enhance)
        )

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        text: Union[str, List[str]],
        enhance: bool=False,
        seed: SeedType=None,
        reference_audio: Optional[AudioType]=None,
        reference_text: Optional[str]=None,
        speed: float=1.0,
        sway_sampling_coef: float=-1.0,
        target_rms: float=0.1,
        cross_fade_duration: float=0.15,
        punctuation_pause_duration: float=0.10,
        num_steps: int=32,
        cfg_strength: float=2.0,
        fix_duration: Optional[float]=None,
        output_format: AUDIO_OUTPUT_FORMAT_LITERAL="wav",
        output_upload: bool=False,
    ) -> AudioResultType:
        """
        Generate speech from text and reference audio.

        :param text: The text to synthesize.
        :param enhance: Whether to enhance the audio using the DeepFilterNet3 model.
        :param seed: The seed to use for random number generation.
        :param reference_audio: The reference audio to use for synthesis.
        :param reference_text: The reference text to use for synthesis.
        :param speed: The speed of the synthesized audio.
        :param sway_sampling_coef: The sampling coefficient for sway sampling.
        :param target_rms: The target RMS value for the synthesized audio.
        :param cross_fade_duration: The duration of the cross-fade between audio chunks.
        :param punctuation_pause_duration: The duration of the pause after punctuation.
        :param num_steps: The number of flow estimation steps.
        :param cfg_strength: The strength of classifier-free guidance.
        :param fix_duration: The fixed duration of the synthesized audio.
        :param output_format: The format of the output audio.
        :param output_upload: Whether to upload the output audio to the configured storage backend, or return the audio data directly.
        :return: The synthesized audio.
        """
        import torch
        if not text:
            raise ValueError("Text is required for synthesis.")

        if seed is not None:
            seed_everything(seed)

        if reference_audio is not None:
            reference_audio, reference_sample_rate = audio_to_bct_tensor(reference_audio, sample_rate=self.sample_rate)
            # clamp at 15s
            max_reference_samples = reference_sample_rate * 15 # type: ignore[operator]
            reference_audio = reference_audio[:, :, :max_reference_samples]
            # Make sure the reference audio is a single channel
            if reference_audio.shape[1] > 1:
                reference_audio = reference_audio.mean(dim=1, keepdim=True)

            if reference_text is None:
                reference_text = self.transcribe(reference_audio, sample_rate=reference_sample_rate)
        else:
            reference_audio, reference_text, reference_sample_rate = self.get_default_reference()

        with torch.inference_mode():
            results = self.synthesize(
                texts=[text] if isinstance(text, str) else text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                reference_sample_rate=reference_sample_rate, # type: ignore[arg-type]
                enhance=enhance,
                seed=seed,
                speed=speed,
                sway_sampling_coef=sway_sampling_coef,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                punctuation_pause_duration=punctuation_pause_duration,
                num_steps=num_steps,
                cfg_strength=cfg_strength,
                fix_duration=fix_duration,
            )

            # This utility method will get the requested format
            return self.get_output_from_audio_result(
                results.unsqueeze(0),
                sample_rate=self.get_sample_rate(enhance=enhance),
                output_format=output_format,
                output_upload=output_upload,
                return_first_item=True
            )
