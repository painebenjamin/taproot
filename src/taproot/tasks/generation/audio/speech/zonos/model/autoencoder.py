# Adapted from https://github.com/Zyphra/Zonos
import math
import torch
import torchaudio # type: ignore[import-untyped]

from typing import List

from transformers.models.dac import DacModel, DacConfig # type: ignore[import-untyped]

__all__ = ["DACAutoencoder"]

class DACAutoencoder(torch.nn.Module):
    """
    A wrapper around the DAC model for audio compression and decompression.
    """
    def __init__(
        self,
        codebook_dim: int=8,
        codebook_loss_weight: float=1.0,
        codebook_size: int=1024,
        commitment_loss_weight: float=0.25,
        decoder_hidden_size: int=1536,
        downsampling_ratios: List[int]=[2,4,8,8],
        encoder_hidden_size: int=64,
        hidden_size: int=1024,
        hop_length: int=512,
        model_type: str="dac",
        n_codebooks: int=9,
        quantizer_dropout: float=0.0,
        sampling_rate: int=44100,
        upsampling_ratios: List[int]=[8,8,4,2],
    ) -> None:
        super().__init__()
        config = DacConfig(
            codebook_dim=codebook_dim,
            codebook_loss_weight=codebook_loss_weight,
            codebook_size=codebook_size,
            commitment_loss_weight=commitment_loss_weight,
            decoder_hidden_size=decoder_hidden_size,
            downsampling_ratios=downsampling_ratios,
            encoder_hidden_size=encoder_hidden_size,
            hidden_size=hidden_size,
            hop_length=hop_length,
            model_type=model_type,
            n_codebooks=n_codebooks,
            quantizer_dropout=quantizer_dropout,
            sampling_rate=sampling_rate,
            upsampling_ratios=upsampling_ratios,
        )
        self.dac = DacModel(config)
        self.dac.eval().requires_grad_(False)

    @property
    def codebook_size(self) -> int:
        """
        :return: the number of codebook entries
        """
        return self.dac.config.codebook_size # type: ignore[no-any-return]

    @property
    def num_codebooks(self) -> int:
        """
        :return: the number of codebooks
        """
        return self.dac.quantizer.n_codebooks # type: ignore[no-any-return]

    @property
    def sampling_rate(self) -> int:
        """
        :return: the sampling rate of the audio
        """
        return self.dac.config.sampling_rate # type: ignore[no-any-return]

    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Preprocesses the audio by resampling it to the model's
        sampling rate and padding it to a multiple of 512.
        """
        wav = torchaudio.functional.resample(wav, sr, 44_100)
        right_pad = math.ceil(wav.shape[-1] / 512) * 512 - wav.shape[-1]
        return torch.nn.functional.pad(wav, (0, right_pad))

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Encodes the audio into a sequence of audio codes.
        """
        return self.dac.encode(wav).audio_codes # type: ignore[no-any-return]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decodes the audio codes into audio.
        """
        return self.dac.decode(audio_codes=codes).audio_values.unsqueeze(1) # type: ignore[no-any-return]
