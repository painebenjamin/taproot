import torch

from typing import Optional
from typing_extensions import Literal

from taproot.modeling.modules.base import Module

__all__ = ["MelSpectrogram"]

class MelSpectrogram(Module):
    """
    MelSpectrogram module. This is a thin wrapper of `torchaudio.transforms.MelSpectrogram`.
    """
    def __init__(
        self,
        filter_length: int=1024,
        hop_length: int=256,
        win_length: int=1024,
        n_mel_channels: int=100,
        target_sample_rate: int=24_000,
        normalize: bool=False,
        power: int=1,
        norm: Optional[Literal["slaney"]]=None,
        center: bool=True,
    ) -> None:
        """
        :param filter_length: The number of filter taps in the filter bank.
        :param hop_length: The number of samples between successive frames.
        :param win_length: The window size.
        :param n_mel_channels: The number of mel filter banks.
        :param target_sample_rate: The target sample rate.
        :param normalize: Whether to normalize the mel spectrogram.
        :param power: The exponent to raise the mel spectrogram.
        :param norm: The normalization method.
        :param center: Whether to pad the signal on both sides.
        """
        try:
            import torchaudio # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("Please install torchaudio to use MelSpectrogram.")
        super().__init__()
        self.num_channels = n_mel_channels
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=filter_length,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=power,
            center=center,
            normalized=normalize,
            norm=norm,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculate the mel spectrogram.

        :param t: The input tensor (B, T) or (B, 1, T).
        :return: The mel spectrogram (B, n_mel_channels, T').
        """
        if t.ndim == 3:
            t = t.squeeze(1)  # 'b 1 nw -> b nw'
        else:
            assert t.ndim == 2, f"The input tensor must be 2D or 3D, got {t.ndim}D."

        if self.device != t.device:
            self.to(t.device) # will move both the dummy buffer and the MelSpectrogram module to the same device

        mel = self.mel_stft(t)
        mel = mel.clamp(min=1e-5).log()
        return mel # type: ignore[no-any-return]
