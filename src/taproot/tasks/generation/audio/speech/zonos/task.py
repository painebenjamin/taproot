from __future__ import annotations

import warnings

from typing import Optional, List, Union, Dict, Tuple, TYPE_CHECKING

from taproot.constants import *
from taproot.tasks.base import Task
from taproot.util import (
    get_seed,
    maybe_use_tqdm,
    trim_silence,
    log_duration,
    get_punctuation_pause_ratio,
    seed_everything,
    equalize_audio,
    pitch_shift_audio,
    rms_normalize_audio,
    audio_to_bct_tensor,
    concatenate_audio,
    normalize_jp_text,
    normalize_text,
    chunk_text,
)

from .pretrained import (
    ZonosHybridModel,
    ZonosTransformerModel,
    ZonosAutoencoderModel,
    ZonosSpeakerEmbeddingModel
)

from taproot.tasks.transformation import DeepFilterNet3Enhancement

if TYPE_CHECKING:
    import torch
    from taproot.hinting import AudioType, AudioResultType, SeedType

__all__ = [
    "ZonosHybridSpeechSynthesis",
    "ZonosTransformerSpeechSynthesis"
]

class ZonosHybridSpeechSynthesis(Task):
    """
    Generate speech from text using the Zonos Hybrid model.
    """

    """Global Task Metadata"""
    task: str = "speech-synthesis"
    model: Optional[str] = "zonos-hybrid"
    default: bool = False
    libraries = [LIBRARY_ESPEAK] # Required libraries
    static_gpu_memory_gb: Optional[float] = 4.04
    static_memory_gb: Optional[float] = 0.15855 # 158.55 MB (torch+espeak)
    pretrained_models = {
        "model": ZonosHybridModel,
        "autoencoder": ZonosAutoencoderModel,
        "speaker_embedding": ZonosSpeakerEmbeddingModel
    }
    optional_component_tasks = {
        "enhance": DeepFilterNet3Enhancement,
    }

    """Authorship Metadata"""
    author = "Zyphra Team"
    author_url = "https://www.zyphra.com/post/beta-release-of-zonos-v0-1"

    """License Metadata"""
    license = LICENSE_APACHE

    """Internal Metadata"""
    sample_rate: int = 44100

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Required packages.
        """
        return {
            "accelerate": ACCELERATE_VERSION_SPEC,
            "causal-conv1d": CAUSAL_CONV1D_VERSION_SPEC,
            "einops": EINOPS_VERSION_SPEC,
            "flash_attn": FLASH_ATTN_VERSION_SPEC,
            "mamba_ssm": MAMBA_SSM_VERSION_SPEC,
            "ninja": NINJA_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
            "phonemizer": PHONEMIZER_VERSION_SPEC,
            "pil": PILLOW_VERSION_SPEC,
            "safetensors": SAFETENSORS_VERSION_SPEC,
            "scipy": SCIPY_VERSION_SPEC,
            "sklearn": SKLEARN_VERSION_SPEC,
            "torch": TORCH_VERSION_SPEC,
            "torchaudio": TORCHAUDIO_VERSION_SPEC,
            "torchvision": TORCHVISION_VERSION_SPEC,
            "transformers": TRANSFORMERS_VERSION_SPEC,
        }

    @property
    def supported_languages(self) -> List[str]:
        """
        Supported languages.
        """
        if not hasattr(self, "_supported_languages"):
            from .model.phonemization import LANGUAGES
            self._supported_languages = LANGUAGES
        return self._supported_languages

    def get_sample_rate(self, enhance: bool=False) -> int:
        """
        Get the sample rate of the model.
        """
        if enhance:
            return self.tasks.enhance.sample_rate # type: ignore[no-any-return]
        return self.sample_rate

    def enhance(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        seed: SeedType=None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Enhance audio using the DeepFilterNet3 model.
        """
        if not DeepFilterNet3Enhancement.is_available():
            warnings.warn("DeepFilterNet3 enhancement is not available.")
            return audio, sample_rate

        result_audio = self.tasks.enhance(
            audio=audio,
            sample_rate=sample_rate,
            seed=seed,
            output_format="float"
        )[0]

        return result_audio, self.tasks.enhance.sample_rate

    def synthesize(
        self,
        texts: List[str],
        seed: int,
        language: str="en-us",
        prefix_audio: Optional[torch.Tensor]=None,
        reference_audio: Optional[torch.Tensor]=None,
        reference_audio_pitch_shift: Optional[Union[str, float]]=-44.99,
        enhance: bool=False,
        cross_fade_duration: float=0.15,
        punctuation_pause_duration: float=0.10,
        cfg_scale: float=2.0,
        target_rms: float=0.1,
        fmax: float=22050.0,
        pitch_std: float=20.0,
        dnsmos: float=4.0,
        vq_score: float=0.78,
        min_p: float=0.15,
        ctc_loss: float=0.0,
        speaker_noised: bool=False,
        speaking_rate: float=15.0,
        emotion_happiness: float=0.3077,
        emotion_sadness: float=0.0256,
        emotion_disgust: float=0.0256,
        emotion_fear: float=0.0256,
        emotion_surprise: float=0.0256,
        emotion_anger: float=0.0256,
        emotion_other: float=0.0256,
        emotion_neutral: float=0.3077,
        skip_speaker: bool=False,
        skip_fmax: bool=False,
        skip_pitch: bool=False,
        skip_dnsmos: bool=False,
        skip_emotion: bool=False,
        skip_vq_score: bool=False,
        skip_speaking_rate: bool=False,
        skip_ctc_loss: bool=False,
    ) -> torch.Tensor:
        """
        Synthesize speech from text and reference audio.
        """
        import torch
        from .model import make_cond_dict

        speaker: Optional[torch.Tensor] = None
        prefix_codes: Optional[torch.Tensor] = None
        audios: List[torch.Tensor] = []
        num_texts = len(texts)
        unconditional_keys = set()
        emotions = [
            emotion_happiness,
            emotion_sadness,
            emotion_disgust,
            emotion_fear,
            emotion_surprise,
            emotion_anger,
            emotion_other,
            emotion_neutral
        ]
        final_sample_rate = self.get_sample_rate(enhance=enhance)

        if reference_audio is not None:
            reference_audio = reference_audio.to(self.device)
            if len(reference_audio.shape) == 3:
                reference_audio = reference_audio[0]
            with log_duration("Extracting speaker embedding"):
                _, speaker = self.pretrained.speaker_embedding(reference_audio)
                assert speaker is not None, "Could not extract speaker embedding."
                speaker = speaker.unsqueeze(0)

        if prefix_audio is not None:
            prefix_audio = prefix_audio.to(self.device)
            prefix_codes = self.pretrained.autoencoder.encode(prefix_audio)

        if skip_speaker or speaker is None:
            unconditional_keys.add("speaker")
        if skip_fmax:
            unconditional_keys.add("fmax")
        if skip_pitch:
            unconditional_keys.add("pitch_std")
        if skip_dnsmos:
            unconditional_keys.add("dnsmos_ovrl")
        if skip_vq_score:
            unconditional_keys.add("vqscore_8")
        if skip_speaking_rate:
            unconditional_keys.add("speaking_rate")
        if skip_ctc_loss:
            unconditional_keys.add("ctc_loss")
        if skip_emotion:
            unconditional_keys.add("emotion")

        for i, text in maybe_use_tqdm(enumerate(texts), desc="Synthesizing Speech", total=num_texts):
            # Re-seed every chunk
            seed_everything(seed)
            conds = make_cond_dict(
                text=text,
                language=language,
                speaker=speaker,
                emotion=emotions,
                fmax=fmax,
                pitch_std=pitch_std,
                speaking_rate=speaking_rate,
                vqscore_8=[vq_score] * 8,
                dnsmos_ovrl=dnsmos,
                speaker_noised=speaker_noised,
                unconditional_keys=unconditional_keys,
                device=self.device,
                dtype=torch.bfloat16
            )
            conditioning = self.pretrained.model.prepare_conditioning(conds)

            codes = self.pretrained.model.generate(
                prefix_conditioning=conditioning,
                audio_prefix_codes=prefix_codes,
                cfg_scale=cfg_scale,
                sampling_params={"min_p": min_p},
            )
            audio = self.pretrained.autoencoder.decode(codes).cpu()[0]
            audio = trim_silence(audio)
            audio = rms_normalize_audio(audio, target_rms)

            if enhance:
                audio, _ = self.enhance(audio, self.sample_rate)

            if (i < num_texts - 1):
                if "speaker" in unconditional_keys:
                    # In order to get a consistent speaker over multiple chunks, there needs to be a speaker embedding.
                    # If the speaker is not provided, we will generate an embedding from the first chunk.
                    if speaker is None:
                        if reference_audio_pitch_shift is not None:
                            reference_audio, _ = pitch_shift_audio(audio, final_sample_rate, reference_audio_pitch_shift)
                        else:
                            reference_audio = audio

                        reference_audio, _ = equalize_audio(reference_audio, final_sample_rate, preset="voice") # type: ignore[arg-type]
                        with log_duration("Extracting speaker embedding"):
                            resampled_audio, _ = audio_to_bct_tensor(
                                reference_audio,
                                sample_rate=final_sample_rate,
                                target_sample_rate=16000
                            )
                            _, speaker = self.pretrained.speaker_embedding(resampled_audio[0].to(self.device))
                            assert speaker is not None, "Could not extract speaker embedding."
                            speaker = speaker.unsqueeze(0)

                    unconditional_keys.remove("speaker")

                pause = get_punctuation_pause_ratio(text)
                if pause > 0:
                    pause_duration = punctuation_pause_duration * pause
                    num_pause_samples = int(pause_duration * final_sample_rate)
                    pause_samples = torch.zeros((1, num_pause_samples)).to(dtype=audio.dtype, device=audio.device)
                    audio = torch.cat([audio, pause_samples], dim=1)

            audios.append(audio)
            self.increment_step()

        return concatenate_audio(
            audios,
            cross_fade_duration=cross_fade_duration,
            sample_rate=final_sample_rate
        )

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        text: Union[str, List[str]],
        language: str="en-us",
        enhance: bool=False,
        seed: SeedType=None,
        prefix_audio: Optional[AudioType]=None,
        equalize_prefix_audio: bool=True,
        enhance_prefix_audio: bool=False,
        reference_audio: Optional[AudioType]=None,
        reference_audio_pitch_shift: Optional[Union[str, float]]=-44.99,
        equalize_reference_audio: bool=True,
        enhance_reference_audio: bool=False,
        cross_fade_duration: float=0.15,
        punctuation_pause_duration: float=0.10,
        cfg_scale: float=2.0,
        max_chunk_length: int=150,
        output_format: AUDIO_OUTPUT_FORMAT_LITERAL="wav",
        output_upload: bool=False,
        target_rms: float=0.1,
        fmax: float=22050.0,
        pitch_std: float=20.0,
        dnsmos: float=4.0,
        vq_score: float=0.78,
        min_p: float=0.15,
        ctc_loss: float=0.0,
        speaker_noised: bool=False,
        speaking_rate: float=15.0,
        emotion_happiness: float=0.3077,
        emotion_sadness: float=0.0256,
        emotion_disgust: float=0.0256,
        emotion_fear: float=0.0256,
        emotion_surprise: float=0.0256,
        emotion_anger: float=0.0256,
        emotion_other: float=0.0256,
        emotion_neutral: float=0.3077,
        skip_speaker: bool=False,
        skip_fmax: bool=False,
        skip_pitch: bool=False,
        skip_dnsmos: bool=False,
        skip_emotion: bool=False,
        skip_vq_score: bool=False,
        skip_speaking_rate: bool=False,
        skip_ctc_loss: bool=False,
    ) -> AudioResultType:
        """
        Generate speech from text and reference audio.

        :param text: The text to synthesize.
        :param language: The language of the text.
        :param enhance: Whether to enhance the audio using the DeepFilterNet3 model.
        :param seed: The seed to use for random number generation.
        :param prefix_audio: The prefix audio to use for synthesis.
        :param reference_audio: The reference audio to use for synthesis.
        :param cross_fade_duration: The duration of the cross-fade between chunks.
        :param punctuation_pause_duration: The duration of the pause after punctuation.
        :param cfg_scale: The scale of classifier-free guidance.
        :param max_chunk_length: The maximum character length of each text chunk.
        :param output_format: The format of the output audio.
        :param output_upload: Whether to upload the output audio to the cloud/local storage.
        :param target_rms: The target root mean square (RMS) value of the audio.
        :param fmax: The maximum frequency of the audio.
        :param pitch_std: The standard deviation of the pitch.
        :param dnsmos: The DNSMOS of the audio (arXiv 2010.15258).
        :param vq_score: The VQScore of the audio (arXiv 2402.16321).
        :param min_p: The minimum probability of generated tokens.
        :param speaker_noised: Whether the speaker is noised.
        :param speaking_rate: The speaking rate of the audio.
        :param emotion_happiness: The happiness emotion value [0.0, 1.0].
        :param emotion_sadness: The sadness emotion value [0.0, 1.0].
        :param emotion_disgust: The disgust emotion value [0.0, 1.0].
        :param emotion_fear: The fear emotion value [0.0, 1.0].
        :param emotion_surprise: The surprise emotion value [0.0, 1.0].
        :param emotion_anger: The anger emotion value [0.0, 1.0].
        :param emotion_other: The other emotion value [0.0, 1.0].
        :param emotion_neutral: The neutral emotion value [0.0, 1.0].
        :param skip_speaker: Whether to skip speaker conditioning.
        :param skip_fmax: Whether to skip fmax conditioning.
        :param skip_pitch: Whether to skip pitch conditioning.
        :param skip_dnsmos: Whether to skip dnsmos conditioning.
        :param skip_emotion: Whether to skip emotion conditioning.
        :param skip_vq_score: Whether to skip VQ score conditioning.
        :param skip_speaking_rate: Whether to skip speaking rate conditioning.
        :return: The synthesized audio.
        """
        import torch

        if not text:
            raise ValueError("Text is required for synthesis.")

        if reference_audio is not None:
            reference_audio, reference_sample_rate = audio_to_bct_tensor(reference_audio)
            reference_audio = trim_silence(reference_audio)
            reference_audio = rms_normalize_audio(reference_audio, target_rms) # type: ignore[arg-type]

            if enhance_reference_audio:
                reference_audio, reference_sample_rate = self.enhance(reference_audio, reference_sample_rate) # type: ignore[arg-type]
            if equalize_reference_audio:
                reference_audio, reference_sample_rate = equalize_audio(reference_audio, reference_sample_rate, preset="voice") # type: ignore[arg-type]
            if reference_audio_pitch_shift:
                reference_audio, reference_sample_rate = pitch_shift_audio(reference_audio, reference_sample_rate, reference_audio_pitch_shift) # type: ignore[arg-type]

            if reference_sample_rate != 16000:
                reference_audio, reference_sample_rate = audio_to_bct_tensor(
                    reference_audio,
                    target_sample_rate=16000
                )

            # Make sure the reference audio is a single channel
            if reference_audio.shape[1] > 1:
                reference_audio = reference_audio.mean(dim=1, keepdim=True)

        if prefix_audio is not None:
            prefix_audio, prefix_sample_rate = audio_to_bct_tensor(prefix_audio)
            prefix_audio = trim_silence(prefix_audio)
            prefix_audio = rms_normalize_audio(prefix_audio, target_rms) # type: ignore[arg-type]

            if enhance_prefix_audio:
                prefix_audio, prefix_sample_rate = self.enhance(prefix_audio, prefix_sample_rate) # type: ignore[arg-type]
            if equalize_prefix_audio:
                prefix_audio, prefix_sample_rate = equalize_audio(prefix_audio, prefix_sample_rate, preset="voice") # type: ignore[arg-type]

            if prefix_sample_rate != self.sample_rate:
                prefix_audio, prefix_sample_rate = audio_to_bct_tensor(
                    prefix_audio,
                    target_sample_rate=self.sample_rate
                )

            # Make sure the prefix audio is a single channel
            if prefix_audio.shape[1] > 1:
                prefix_audio = prefix_audio.mean(dim=1, keepdim=True)

        texts = [text] if isinstance(text, str) else text
        text_chunks = []

        for text in texts:
            if "ja" in language:
                text = normalize_jp_text(text)
            else:
                text = normalize_text(text)
            text_chunks.extend(chunk_text(text, max_length=max_chunk_length))

        self.num_steps = len(text_chunks)
        self.step = 0

        with torch.inference_mode():
            results = self.synthesize(
                texts=text_chunks,
                language=language,
                seed=get_seed(seed),
                prefix_audio=prefix_audio,
                reference_audio=reference_audio,
                reference_audio_pitch_shift=reference_audio_pitch_shift,
                enhance=enhance,
                cross_fade_duration=cross_fade_duration,
                punctuation_pause_duration=punctuation_pause_duration,
                cfg_scale=cfg_scale,
                target_rms=target_rms,
                fmax=fmax,
                pitch_std=pitch_std,
                dnsmos=dnsmos,
                vq_score=vq_score,
                min_p=min_p,
                ctc_loss=ctc_loss,
                speaker_noised=speaker_noised,
                speaking_rate=speaking_rate,
                emotion_happiness=emotion_happiness,
                emotion_sadness=emotion_sadness,
                emotion_disgust=emotion_disgust,
                emotion_fear=emotion_fear,
                emotion_surprise=emotion_surprise,
                emotion_anger=emotion_anger,
                emotion_other=emotion_other,
                emotion_neutral=emotion_neutral,
                skip_speaker=skip_speaker,
                skip_fmax=skip_fmax,
                skip_pitch=skip_pitch,
                skip_dnsmos=skip_dnsmos,
                skip_emotion=skip_emotion,
                skip_vq_score=skip_vq_score,
                skip_speaking_rate=skip_speaking_rate,
                skip_ctc_loss=skip_ctc_loss
            )

            # This utility method will get the requested format
            return self.get_output_from_audio_result(
                results.unsqueeze(0),
                sample_rate=self.get_sample_rate(enhance=enhance),
                output_format=output_format,
                output_upload=output_upload,
                return_first_item=True
            )

class ZonosTransformerSpeechSynthesis(ZonosHybridSpeechSynthesis):
    """
    Generate speech from text using the Zonos Transformer model.
    """
    model: Optional[str] = "zonos-transformer"
    pretrained_models = {
        "model": ZonosTransformerModel,
        "autoencoder": ZonosAutoencoderModel,
        "speaker_embedding": ZonosSpeakerEmbeddingModel
    }
