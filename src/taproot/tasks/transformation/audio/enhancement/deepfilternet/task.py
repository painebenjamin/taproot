from __future__ import annotations

import math

from typing import Optional, Dict, Tuple, TYPE_CHECKING

from taproot.constants import *
from taproot.util import (
    audio_to_bct_tensor,
    is_multi_audio,
    seed_everything,
    tensor_chunk_iterator,
)
from taproot.tasks.base import Task

from .pretrained import DeepFilterNet3Model

if TYPE_CHECKING:
    from torch import Tensor
    from taproot.hinting import AudioType, AudioResultType, SeedType
    from libdf import DF # type: ignore[import-untyped,import-not-found,unused-ignore]
    from df.deepfilternet3 import DfNet # type: ignore[import-untyped,import-not-found,unused-ignore]

__all__ = ["DeepFilterNet3Enhancement"]

class DeepFilterNet3Enhancement(Task):
    """
    A task for enhancing audio using the DeepFilterNet3 model.
    """

    """Global Task Metadata"""
    task = "speech-enhancement"
    model = "deep-filter-net-v3"
    default = True
    display_name = "DeepFilterNet V3 Speech Enhancement"
    pretrained_models = {"model": DeepFilterNet3Model}
    static_memory_gb = 0.06072 # 60 MB
    static_gpu_memory_gb = 0.00876 # 8 MB

    """Authorship Metadata"""
    author_url = "https://arxiv.org/abs/2305.08227"
    author = "Hendrick Schröter"
    author_additional = ["Tobias Rosenkranz", "Alberto N. Escalante-B", "Andreas Maier"]
    author_journal = "INTERSPEECH"
    author_journal_year = 2023
    author_journal_title = "DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement"

    """License Metadata"""
    license = LICENSE_APACHE

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Required packages.
        """
        return {
            "df": ">=0.5",
            "torch": TORCH_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
            "torchvision": TORCHVISION_VERSION_SPEC,
            "transformers": TRANSFORMERS_VERSION_SPEC,
            "torchaudio": TORCHAUDIO_VERSION_SPEC,
            "librosa": LIBROSA_VERSION_SPEC,
            "av": PYAV_VERSION_SPEC,
            "safetensors": SAFETENSORS_VERSION_SPEC,
            "accelerate": ACCELERATE_VERSION_SPEC,
        }

    """Internal Task Attributes"""

    @property
    def max_samples(self) -> int:
        """
        The maximum number of samples that can be processed in a single call.
        """
        return 48000 * 60 * 12 # 12 minutes @ 48 kHz

    @property
    def df_net(self) -> DfNet:
        """
        Add mapped modules as properties for convenience and type hinting.
        """
        return self.pretrained.model

    @property
    def df_state(self) -> DF:
        """
        Get the state of the DeepFilterNet3 model.
        """
        return DeepFilterNet3Model.get_df_state()

    """Private Methods"""

    def _get_norm_alpha(self) -> float:
        """
        Get the normalization alpha for the given type.
        """
        hop_size = self.df_state.hop_size()
        sr = self.df_state.sr()
        dt = hop_size / sr
        alpha = math.exp(-dt)
        precision = 3
        a = 1.0
        while a >= 1.0:
            a = round(alpha, precision)
            precision += 1
        return a

    def _as_real(self, x: Tensor) -> Tensor:
        """
        Convert a complex tensor to a real tensor.
        """
        import torch
        if not torch.is_complex(x):
            return x
        return torch.view_as_real(x)

    def _as_complex(self, x: Tensor) -> Tensor:
        """
        Convert a real tensor to a complex tensor.
        """
        import torch
        if torch.is_complex(x):
            return x
        if x.shape[-1] != 2:
            raise ValueError(f"Last dimension need to be of length 2 (re + im), but got {x.shape}")
        if x.stride(-1) != 1:
            x = x.contiguous()
        return torch.view_as_complex(x)

    def _get_df_features(
        self,
        audio: Tensor,
        nb_df: int=96,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get the DeepFilterNet features from the audio tensor.
        """
        import torch
        from libdf import erb_norm, erb, unit_norm
        spec = self.df_state.analysis(audio.numpy())  # [C, Tf] -> [C, Tf, F]
        a = self._get_norm_alpha()
        erb_fb = self.df_state.erb_widths()
        erb_feat = torch.as_tensor(erb_norm(erb(spec, erb_fb), a)).unsqueeze(1)
        spec_feat = self._as_real(torch.as_tensor(unit_norm(spec[..., :nb_df], a)).unsqueeze(1))
        spec_t = self._as_real(torch.as_tensor(spec).unsqueeze(1))
        spec_t = spec_t.to(self.device)
        erb_feat = erb_feat.to(self.device)
        spec_feat = spec_feat.to(self.device)
        return spec_t, erb_feat, spec_feat

    def _enhance(
        self,
        audio: Tensor,
        pad: bool=True,
        atten_lim_db: Optional[float]=None
    ) -> Tensor:
        """Enhance a single audio given a preloaded model and DF state.

        Args:
            audio (Tensor): Time domain audio of shape [C, T]. Sampling rate needs to match to `model` and `df_state`.
            pad (bool): Pad the audio to compensate for delay due to STFT/ISTFT.
            atten_lim_db (float): An optional noise attenuation limit in dB. E.g. an attenuation limit of
                12 dB only suppresses 12 dB and keeps the remaining noise in the resulting audio.

        Returns:
            enhanced audio (Tensor): If `pad` was `False` of shape [C, T'] where T'<T slightly delayed due to STFT.
                If `pad` was `True` it has the same shape as the input.
        """
        import torch
        import torch.nn.functional as F
        bs = audio.shape[0]
        if hasattr(self.df_net, "reset_h0"):
            self.df_net.reset_h0(batch_size=bs, device=self.device)

        orig_len = audio.shape[-1]
        n_fft, hop = 0, 0
        if pad:
            n_fft, hop = self.df_state.fft_size(), self.df_state.hop_size()
            # Pad audio to compensate for the delay due to the real-time STFT implementation
            audio = F.pad(audio, (0, n_fft))

        spec, erb_feat, spec_feat = self._get_df_features(audio)
        enhanced = self.df_net(spec, erb_feat, spec_feat)[0].cpu()
        enhanced = self._as_complex(enhanced.squeeze(1))
        if atten_lim_db is not None and abs(atten_lim_db) > 0:
            lim = 10 ** (-abs(atten_lim_db) / 20)
            enhanced = self._as_complex(spec.squeeze(1).cpu()) * lim + enhanced * (1 - lim)
        audio = torch.as_tensor(self.df_state.synthesis(enhanced.numpy()))
        if pad:
            # The frame size is equal to p.hop_size. Given a new frame, the STFT loop requires e.g.
            # ceil((n_fft-hop)/hop). I.e. for 50% overlap, then hop=n_fft//2
            # requires 1 additional frame lookahead; 75% requires 3 additional frames lookahead.
            # Thus, the STFT/ISTFT loop introduces an algorithmic delay of n_fft - hop.
            assert n_fft % hop == 0  # This is only tested for 50% and 75% overlap
            d = n_fft - hop
            audio = audio[:, d : orig_len + d]
        return audio

    def _process(self, audio: Tensor) -> Tensor:
        """
        Process the audio tensor.
        """
        import torch
        enhanced = [
            self._enhance(chunk)
            for chunk in tensor_chunk_iterator(
                audio,
                dim=1,
                size=self.max_samples,
            )
        ]
        return torch.cat(enhanced, dim=1)

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        audio: AudioType,
        seed: SeedType=None,
        sample_rate: Optional[int]=None,
        output_format: AUDIO_OUTPUT_FORMAT_LITERAL="wav",
        output_upload: bool=False,
     ) -> AudioResultType:
        """
        Suppress noise and enhance speech in audio using the DeepFilterNet3 model.
        """
        import torch
        if seed is not None:
            seed_everything(seed)
        with torch.inference_mode():
            audios, sr = audio_to_bct_tensor(
                audio,
                sample_rate=sample_rate,
                target_sample_rate=self.df_state.sr(),
            )
            results = torch.stack([
                self._process(audio)
                for audio in audios.float()
            ])

        return self.get_output_from_audio_result(
            results,
            sample_rate=self.df_state.sr(),
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=not is_multi_audio(audio)
        )
