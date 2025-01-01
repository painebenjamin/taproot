from __future__ import annotations

import re

from typing import Optional, Dict, Any, cast, TYPE_CHECKING
from typing_extensions import Literal

from taproot.util import audio_to_bct_tensor, seed_everything

from taproot.constants import *
from taproot.tasks.base import Task

# Pretrained model
from .pretrained import XTTS2Model

# Component tasks
from taproot.tasks.transformation import DeepFilterNet3Enhancement

if TYPE_CHECKING:
    import torch
    from taproot.hinting import AudioType, AudioResultType, SeedType
    from .model import XTTS2

__all__ = ["XTTS2SpeechSynthesis"]

class XTTS2SpeechSynthesis(Task):
    """
    Speech synthesis task using the XTTS2 model.
    """

    """Global Task Metadata"""
    task: str = "speech-synthesis"
    model: Optional[str] = "xtts-v2"
    default: bool = True
    display_name = "XTTS2 Speech Synthesis"
    pretrained_models = {"model": XTTS2Model}
    optional_component_tasks = {"enhance": DeepFilterNet3Enhancement}
    # Overrides
    measure_memory = False # Memory measurement breaks the task (for some reason)

    """Authorship Metadata"""
    author = "Coqui AI"
    author_url = "https://coqui.ai/blog/tts/open_xtts"
    author_journal = "Coqui AI Blog"
    author_journal_year = 2023
    author_journal_title = "XTTS: Open Model Release Announcement"

    """Licensing Metadata"""
    license = LICENSE_MPL

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Required packages.
        """
        return {
            "pil": PILLOW_VERSION_SPEC,
            "torch": TORCH_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
            "scipy": SCIPY_VERSION_SPEC,
            "pandas": PANDAS_VERSION_SPEC,
            "torchvision": TORCHVISION_VERSION_SPEC,
            "torchaudio": TORCHAUDIO_VERSION_SPEC,
            "accelerate": ACCELERATE_VERSION_SPEC,
            "safetensors": SAFETENSORS_VERSION_SPEC,
            "librosa": LIBROSA_VERSION_SPEC,
            "einops": EINOPS_VERSION_SPEC,
            "sklearn": SKLEARN_VERSION_SPEC,
            "tts": "~=0.22",
            "spacy[ja]": "~=3.7.4",
            "transformers": "==4.42.4", # Freeze version
        }

    @classmethod
    def required_static_gpu_memory_gb(cls) -> Optional[float]:
        """
        The amount of GPU memory required for the task in GB.
        """
        return 1.91

    """Internal Task Attributes"""

    @property
    def xtts(self) -> XTTS2:
        """
        Add mapped modules as properties for convenience and type hinting.
        """
        return self.pretrained.model # type: ignore[no-any-return]

    """Override Properties"""

    @property
    def last_intermediate(self) -> Any:
        """
        Override last intermediates to concatenate instead of return the last.
        """
        if self.intermediates:
            import torch
            return torch.cat(self.intermediates)
        return None

    """Public Task Methods"""

    def get_sample_rate(self, enhance: bool=False) -> int:
        """
        Get the sample rate of the model.
        """
        if enhance:
            return self.tasks.enhance.df_state.sr() # type: ignore[no-any-return]
        return self.xtts.config.audio.sample_rate # type: ignore[no-any-return]

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
            sample_rate=self.get_sample_rate(enhance=False), # get unenhanced sample rate
            seed=seed,
            output_format="float"
        )[0]

    def format_prompt(
        self,
        text: str
    ) -> str:
        """
        Formats a prompt for the XTTS model.

        There are some quirks with how the model pronounces certain formats of text.
        This method attempts to fix some of these issues.
        """
        # Look for numbers that lead with a '-' and replace with 'negative'
        text = re.sub(r"([^a-zA-Z0-9])-(\d+)", r"\1negative \2", text)
        # Look for degrees F, replaces with 'Fahrenheit'
        text = re.sub(r"(\d+)\s*°\s*F", r"\1 degrees Fahrenheit", text)
        # Look for degrees C, replaces with 'Celsius'
        text = re.sub(r"(\d+)\s*°\s*C", r"\1 degrees Celsius", text)
        # Re-order currency when using abbreviations, like $1M -> 1 million dollars (instead of dollars 1 million or 1 dollars million)
        # Thousand
        text = re.sub(r"\$(\d+)\s*[kK]", r"\1 thousand dollars", text)
        # Million
        text = re.sub(r"\$(\d+)\s*[mM]", r"\1 million dollars", text)
        # Billion
        text = re.sub(r"\$(\d+)\s*[bB]", r"\1 billion dollars", text)
        # Replace common abbreviations
        text = re.sub(r"(\d+)\W*[mM][pP][hH]\W", r"\1 miles per hour", text)
        text = re.sub(r"(\d+)\W*[kK][pP][hH]\W", r"\1 kilometers per hour", text)
        text = re.sub(r"(\d+)\W*[kK][gG]\W", r"\1 kilograms", text)
        text = re.sub(r"(\d+)\W*[lL][bB][sS]?\W", r"\1 pounds", text)
        text = re.sub(r"(\d+)\W*[fF][tT]\W", r"\1 feet", text)
        text = re.sub(r"(\d+)\W*[mM][\W]", r"\1 meters", text)
        text = re.sub(r"(\d+)\W*[cC][mM]\W", r"\1 centimeters", text)
        text = re.sub(r"(\d+)\W*[iI][nN]\W", r"\1 inches", text)
        text = re.sub(r"(\d+)\W*[mM][iI]\W", r"\1 miles", text)
        text = re.sub(r"(\d+)\W*[kK][mM]\W", r"\1 kilometers", text)
        # Replace ampersands with 'and'
        text = re.sub(r"&", " and ", text)
        # Replace the 'at' symbol with 'at'
        text = re.sub(r"@", " at ", text)
        # Replace newlines and carriage returns with spaces
        text = re.sub(r"([\n\r])", " ", text)
        # Replace multiple spaces with a single space
        text = re.sub(r"\s+", " ", text)
        # Remove uncommon punctuation which can be interpreted as other languages
        text = re.sub(r"[^a-zA-Z0-9,.:;\ \-'\"\/()!?]", "", text)

        return text.strip()

    def synthesize(
        self,
        text: str,
        language: str="en",
        reference_audio: Optional[AudioType]=None,
        speaker_id: Optional[str]=None,
        enhance: bool=False,
        stream: bool=False,
        stream_chunk_size: int=20,
        seed: SeedType=None,
        speed: Optional[float]=None,
    ) -> torch.Tensor:
        """
        Synthesize audio from text.
        """
        import torch
        if reference_audio is not None:
            speaker_wav, _ = audio_to_bct_tensor(
                reference_audio,
                sample_rate=self.get_sample_rate(enhance=False)
            )
        else:
            speaker_wav = None

        formatted_text = self.format_prompt(text)
        if not formatted_text:
            # After adjusting text there's nothing to say, return 1 second of silence
            return torch.zeros((self.get_sample_rate(enhance=enhance),), dtype=torch.float32)

        audio = self.xtts(
            self.format_prompt(text),
            language=language,
            speaker_wav=speaker_wav,
            speaker_id=speaker_id,
            stream=stream,
            stream_chunk_size=stream_chunk_size,
            speed=1.0 if speed is None else speed
        )

        if stream:
            for waveform in audio:
                if enhance:
                    waveform = self.enhance(waveform.cpu(), seed=seed) # type: ignore[union-attr]
                self.add_intermediate(waveform)
            return self.last_intermediate # type: ignore[no-any-return]
        else:
            waveform = cast(torch.Tensor, audio["wav"]) # type: ignore[index]
            if enhance:
                waveform = self.enhance(waveform, seed=seed)
            return waveform

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        text: str,
        enhance: bool=False,
        seed: SeedType=None,
        speed: Optional[float]=None,
        language: Literal["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]="en",
        reference_audio: Optional[AudioType]=None,
        speaker_id: Optional[str]=None,
        stream: bool=False,
        stream_chunk_size: int=20,
        output_format: AUDIO_OUTPUT_FORMAT_LITERAL="wav",
        output_upload: bool=False,
    ) -> AudioResultType:
        """
        Generate audio from text.

        :param text: The text to synthesize.
        :param enhance: Whether to enhance the audio using the DeepFilterNet3 model.
        :param seed: The seed to use for random number generation.
        :param speed: The speed factor to apply to the audio.
        :param language: The language of the text.
        :param reference_audio: The reference audio to use for speaker adaptation.
        :param speaker_id: The speaker ID to use for speaker adaptation when reference_audio is not provided.
        :param stream: Whether to stream the audio in chunks.
        :param stream_chunk_size: The size of the chunks to stream (in tokens).
        :param output_format: The format of the output audio.
        :param output_upload: Whether to upload the output audio to the configured storage backend, or return the audio data directly.
        :return: The synthesized audio.
        """
        import torch
        if seed is not None:
            seed_everything(seed)

        with torch.inference_mode():
            results = self.synthesize(
                text,
                language=language,
                reference_audio=reference_audio,
                speaker_id=speaker_id,
                enhance=enhance,
                stream=stream,
                stream_chunk_size=stream_chunk_size,
                seed=seed,
                speed=speed
            ).unsqueeze(0) # Add batch dimension

            # This utility method will get the requested format
            return self.get_output_from_audio_result(
                results,
                sample_rate=self.get_sample_rate(enhance=enhance),
                output_format=output_format,
                output_upload=output_upload,
                return_first_item=True
            )
