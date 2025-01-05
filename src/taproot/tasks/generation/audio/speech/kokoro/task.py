from __future__ import annotations

from typing import Optional, Union, List, Dict, Any, TYPE_CHECKING

from taproot.constants import *
from taproot.util import (
    logger,
    get_seed,
    concatenate_audio,
    seed_everything,
    normalize_text,
    chunk_text
)

from taproot.tasks.base import Task

# Pretrained model
from .pretrained import KokoroV019Model

# Components (sub-tasks)
from taproot.tasks.transformation import DeepFilterNet3Enhancement

if TYPE_CHECKING:
    import torch
    from .model import KokoroModel
    from taproot.hinting import AudioResultType, SeedType

__all__ = ["KokoroSpeechSynthesis"]

class KokoroSpeechSynthesis(Task):
    """
    Speech synthesis task using the Kokoro model.
    """

    """Global Task Metadata"""
    task = "speech-synthesis"
    model = "kokoro"
    default = False
    display_name = "Kokoro Speech Synthesis"
    libraries = [LIBRARY_ESPEAK] # Required libraries
    pretrained_models = {"model": KokoroV019Model}
    optional_component_tasks = {"enhance": DeepFilterNet3Enhancement}
    static_memory_gb = 0.14136 # 141.36 MB
    static_gpu_memory_gb = 0.33254 # 332.54 MB

    """Authorship Metadata"""
    author = "@rzvzn"
    author_additional = ["Yinghao Aaron Li", "Cong Han", "Vinay S. Raghavan", "Gavin Mischler", "Nima Mesgarani"]
    author_url = "https://huggingface.co/hexgrad/Kokoro-82M"

    """License Metadata"""
    license = LICENSE_APACHE

    """Task-Specific Metadata"""
    sample_rate: int = 24000

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
            "phonemizer": PHONEMIZER_VERSION_SPEC,
        }

    """Internal Task Attributes"""

    @property
    def kokoro(self) -> KokoroModel:
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
            # No cross-fading for intermediates
            return concatenate_audio(self.intermediates)
        return None

    """Public Task Methods"""

    def get_sample_rate(self, enhance: bool=False) -> int:
        """
        Get the sample rate of the model.
        """
        if enhance:
            return self.tasks.enhance.df_state.sr() # type: ignore[no-any-return]
        return self.sample_rate

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

    def synthesize(
        self,
        texts: List[str],
        voice: Optional[str]=None,
        speed: float=1.0,
        enhance: bool=False,
        seed: SeedType=None,
        cross_fade_duration: float=0.15,
        punctuation_pause_duration: float=0.10,
        stream: bool=False,
    ) -> torch.Tensor:
        """
        Synthesize audio from text.
        """
        import torch

        audios: List[torch.Tensor] = []
        num_texts = len(texts)

        for i, text in enumerate(texts):
            text = normalize_text(text)
            text_chunks = chunk_text(text)
            num_text_chunks = len(text_chunks)

            for j, text_chunk in enumerate(text_chunks):
                logger.debug(
                    f"Generating audio from text chunk {j + 1} of {num_text_chunks} " +
                    f"for text {i + 1} of {num_texts}, text chunk: {text_chunk}"
                )
                audio = self.kokoro.generate(
                    voice=voice,
                    text=text_chunk,
                    speed=speed,
                )

                audio = audio.squeeze().cpu() # type: ignore[union-attr]

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
        voice: Optional[str]=None,
        enhance: bool=False,
        seed: SeedType=None,
        speed: float=1.0,
        stream: bool=False,
        cross_fade_duration: float=0.15,
        punctuation_pause_duration: float=0.10,
        output_format: AUDIO_OUTPUT_FORMAT_LITERAL="wav",
        output_upload: bool=False,
    ) -> AudioResultType:
        """
        Generate speech from text and reference audio.

        :param text: The text to synthesize.
        :param voice: The voice to use for synthesis. Default is random.
        :param enhance: Whether to enhance the audio using the DeepFilterNet3 model.
        :param seed: The seed to use for random number generation.
        :param cross_fade_duration: The duration of the cross-fade between audio chunks.
        :param punctuation_pause_duration: The duration of the pause after punctuation.
        :param output_format: The format of the output audio.
        :param output_upload: Whether to upload the output audio to the configured storage backend, or return the audio data directly.
        :return: The synthesized audio.
        """
        import torch
        if not text:
            raise ValueError("Text is required for synthesis.")

        seed_everything(get_seed(seed))

        with torch.inference_mode():
            results = self.synthesize(
                texts=[text] if isinstance(text, str) else text,
                voice=voice,
                enhance=enhance,
                seed=seed,
                speed=speed,
                stream=stream,
                cross_fade_duration=cross_fade_duration,
                punctuation_pause_duration=punctuation_pause_duration,
            )

        # This utility method will get the requested format
        return self.get_output_from_audio_result(
            results.unsqueeze(0),
            sample_rate=self.get_sample_rate(enhance=enhance),
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=True
        )
