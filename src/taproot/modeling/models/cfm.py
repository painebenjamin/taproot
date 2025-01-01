from __future__ import annotations

import torch
import torch.nn as nn

from typing import Dict, Optional, Tuple, List, Union, Callable

from torch import Tensor, Generator
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint # type: ignore[import-not-found,import-untyped,unused-ignore,attr-defined]

from taproot.util import (
    mask_from_seq_lengths,
    mask_from_frac_lengths,
)

from ..modules import Module, MelSpectrogram
from ..architectures import DiT

class ConditionalFlowMatching(Module):
    """
    Conditional Flow Matching (CFM) module.

    Proposed by Yushen Chen et. al. in "F5-TTS: A Fairytaler that Fakes Fluent
and Faithful Speech with Flow Matching" (2024).

    :see https://arxiv.org/pdf/2410.06885:
    """
    def __init__(
        self,
        transformer: DiT, # TODO: Support UnetT
        sigma: float=0.0,
        ode_solver: Optional[str]="euler",
        audio_drop_rate: float=0.3,
        cond_drop_rate: float=0.2,
        num_channels: Optional[int]=None,
        mel_spec: Optional[nn.Module]=None,
        frac_lengths_mask: Tuple[float, float]=(0.7, 1.0),
        vocab_char_map: Optional[Dict[str, int]]=None,
    ) -> None:
        """
        :param transformer: Transformer module.
        :param sigma: Standard deviation of the Gaussian noise.
        :param ode_solver: ODE solver to use.
        :param audio_drop_rate: Dropout rate for audio.
        :param cond_drop_rate: Dropout rate for condition.
        :param num_channels: Number of channels.
        :param mel_spec: Mel spectrogram module.
        :param frac_lengths_mask: Fractional lengths mask.
        :param vocab_char_map: Vocabulary character map.
        """
        super().__init__()
        self.frac_lengths_mask = frac_lengths_mask

        # Spectrogram
        self.mel_spec = mel_spec if mel_spec is not None else MelSpectrogram()
        self.num_channels = num_channels if num_channels is not None else self.mel_spec.num_channels

        # Transformer
        self.transformer = transformer

        # Classifier-free Guidance
        self.audio_drop_rate = audio_drop_rate
        self.cond_drop_rate = cond_drop_rate

        # Sampling
        self.sigma = sigma
        self.ode_solver = ode_solver
        self.vocab_char_map = vocab_char_map

        # Compatibility with F5-TTS
        self.register_buffer("step", torch.tensor(0))
        self.register_buffer("initted", torch.tensor(True))

    @classmethod
    def dit(
        cls,
        dim: int,
        depth: int=8,
        heads: int=8,
        dim_head: int=64,
        dropout: float=0.1,
        ff_mult: int=4,
        text_num_embeds: int=256,
        mel_dim: int=100,
        text_dim: Optional[int]=None,
        conv_layers: Optional[int]=None,
        long_skip_connection: bool=False,
        sigma: float=0.0,
        ode_solver: Optional[str]="euler",
        audio_drop_rate: float=0.3,
        cond_drop_rate: float=0.2,
        num_channels: Optional[int]=None,
        mel_spec: Optional[nn.Module]=None,
        frac_lengths_mask: Tuple[float, float]=(0.7, 1.0),
        vocab_char_map: Optional[Dict[str, int]]=None,
    ) -> ConditionalFlowMatching:
        """
        Initializes a Conditional Flow Matching (CFM) module with a DiT transformer.

        :param dim: Dimension of the transformer.
        :param depth: Depth of the transformer.
        :param heads: Number of heads.
        :param dim_head: Dimension of the head.
        :param dropout: Dropout rate.
        :param ff_mult: Feed-forward multiplier.
        :param text_num_embeds: Number of text embeddings.
        :param mel_dim: Dimension of the mel spectrogram.
        :param text_dim: Dimension of the text embeddings.
        :param conv_layers: Number of convolutional layers.
        :param long_skip_connection: Whether to use long skip connections.
        """
        transformer = DiT(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            ff_mult=ff_mult,
            text_num_embeds=text_num_embeds if not vocab_char_map else len(vocab_char_map),
            mel_dim=mel_dim,
            text_dim=text_dim,
            conv_layers=conv_layers,
            long_skip_connection=long_skip_connection,
        )

        return cls(
            transformer=transformer,
            sigma=sigma,
            ode_solver=ode_solver,
            audio_drop_rate=audio_drop_rate,
            cond_drop_rate=cond_drop_rate,
            num_channels=num_channels,
            mel_spec=mel_spec,
            frac_lengths_mask=frac_lengths_mask,
            vocab_char_map=vocab_char_map,
        )

    @property
    def dim(self) -> int:
        """
        Returns the dimension of the transformer.
        """
        return self.transformer.dim

    @property
    def ema_model(self) -> nn.Module:
        """
        Maps the property `ema_model` to self for compatibility with
        the official f5-tts implementation state dict.
        """
        return self

    def ema_model_state_dict(self) -> nn.Module:
        """
        Maps the property `ema_model_state_dict` to self for compatibility with
        the official f5-tts implementation state dict.
        """
        return self

    def texts_to_tensor(self, texts: List[str]) -> Tensor:
        """
        Converts a list of strings to a tensor.

        :param strings: List of strings.
        :return: Tensor.
        """
        if self.vocab_char_map is None:
            # Tokenize with utf-8
            list_tensor = [
                torch.tensor([*bytes(t, "utf-8")])
                for t in texts
            ]
        else:
            list_tensor = [
                torch.tensor([self.vocab_char_map.get(c, 0) for c in t])
                for t in texts
            ]
        return pad_sequence(list_tensor, padding_value=-1, batch_first=True)

    def forward(
        self,
        x: Tensor, # raw wav or mel-scale spectrogram
        text: Union[Tensor, List[str], str], # text embeddings
        seq_lengths: Optional[Tensor]=None,
        generator: Optional[Generator]=None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param x: Input tensor.
        :param text: Text tensor or list of strings.
        :param seq_lengths: Lengths tensor. Optional.
        :param generator: Generator. Optional.
        """
        if x.ndim == 1:
            # Raw waveform, no channel dimension
            x = x.unsqueeze(0)

        if x.ndim == 2:
            # Raw waveform
            x = self.mel_spec(x).permute(0, 2, 1)

        b, s, c = x.shape

        if isinstance(text, str):
            text = [text]

        if isinstance(text, list):
            text = self.texts_to_tensor(text)

        assert isinstance(text, Tensor)
        assert text.size(0) == b, "Text and audio batch sizes must match."

        if seq_lengths is None:
            seq_lengths = torch.full((b,), s, device=self.device)

        mask = mask_from_seq_lengths(seq_lengths, length=s)

        # Get a random span to mask out for conditional training
        frac_lengths = torch.zeros((b,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(seq_lengths, frac_lengths)
        rand_span_mask &= mask

        x_1 = x
        x_0 = torch.randn(
            *x_1.shape,
            device=x_1.device,
            dtype=x_1.dtype,
            generator=generator
        )
        time = torch.rand((b,), dtype=x.dtype, device=x.device, generator=generator)

        # Sample x_t
        t = time.unsqueeze(-1).unsqueeze(-1)
        phi = (1 - t) * x_0 + t * x_1
        flow = x_1 - x_0

        # Only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x_1), x_1)

        # Transformer and cfg training with drop rate
        drop_rand = torch.randn(2, device=x.device, generator=generator)

        if drop_rand[0].item() < self.cond_drop_rate:
            # Drop both audio and text
            drop_audio_cond = True
            drop_text = True
        else:
            drop_audio_cond = drop_rand[1].item() < self.cond_drop_rate
            drop_text = False

        pred = self.transformer(
            x=phi,
            cond=cond,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
        )

        loss = torch.nn.functional.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred

    @torch.no_grad()
    def sample(
        self,
        cond: Tensor,
        text: Union[Tensor, List[str], str],
        duration: Union[Tensor, int],
        seq_lengths: Optional[Tensor]=None,
        steps: int=32,
        cfg_strength: float=1.0,
        sway_sampling_coef: Optional[float]=None,
        seed: Optional[int]=None,
        max_duration: int=4096,
        vocoder: Optional[Callable[[Tensor], Tensor]]=None,
        no_ref_audio: bool=False,
        duplicate_test: bool=False,
        t_inter: float=0.1,
        edit_mask: Optional[Tensor]=None
    ) -> Tuple[Tensor, Tensor]:
        """
        Performs sampling.

        :param cond: Condition tensor.
        :param text: Text tensor or list of strings.
        :param duration: Duration tensor.
        :param seq_lengths: Lengths tensor. Optional.
        :param steps: Number of steps.
        :param cfg_strength: classifier-free guidance strength.
        :param sway_sampling_coef: Sway sampling coefficient.
        :param seed: Random seed.
        :param max_duration: Maximum duration.
        :param vocoder: Vocoder.
        :param no_ref_audio: Whether to use reference audio.
        :param duplicate_test: Whether to duplicate the test.
        :param t_inter: Time interpolation.
        :param edit_mask: Edit mask.
        :
        """
        cond = cond.to(device=self.device, dtype=self.dtype)
        if cond.ndim == 1:
            # raw waveform, no channel dimension
            cond = cond.unsqueeze(0)

        if cond.ndim == 2:
            # raw waveform
            cond = self.mel_spec(cond).permute(0, 2, 1)

        b, s, c = cond.shape

        if isinstance(text, str):
            text = [text]

        if isinstance(text, list):
            text = self.texts_to_tensor(text)

        assert isinstance(text, Tensor)
        text = text.to(device=self.device)

        if seq_lengths is None:
            seq_lengths = torch.full((b,), s, device=self.device)

        text_lengths = (text != -1).sum(dim=1)
        seq_lengths = torch.maximum(text_lengths, seq_lengths)

        cond_mask = mask_from_seq_lengths(seq_lengths)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((b,), duration, device=self.device, dtype=torch.long)

        duration = torch.maximum(seq_lengths + 1, duration) # Add one token to be sure something is generated
        duration = duration.clamp(max=max_duration)

        max_duration = int(duration.amax().item())

        if duplicate_test:
            test_cond = torch.nn.functional.pad(
                cond,
                (0, 0, s, max_duration - 2 * s),
                value=0.0
            )

        cond = torch.nn.functional.pad(
            cond,
            (0, 0, 0, max_duration - s),
            value=0.0
        )
        cond_mask = torch.nn.functional.pad(
            cond_mask,
            (0, max_duration - cond_mask.size(-1)),
            value=False
        )
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask,
            cond,
            torch.zeros_like(cond)
        )

        if b > 1:
            mask = mask_from_seq_lengths(duration)
        else:
            mask = None

        if no_ref_audio:
            cond = torch.zeros_like(cond)

        # ODE solver
        def ode_solver(t: Tensor, x: Tensor) -> Tensor:
            """
            :param t: Time tensor.
            :param x: Input tensor.
            """
            pred = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=False,
                drop_text=False
            )
            if cfg_strength < 1e-5: 
                # Disable classifier-free guidance
                return pred # type: ignore[no-any-return]

            # Perform classifier-free guidance
            null_pred = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=True,
                drop_text=True
            )
            return pred + (pred - null_pred) * cfg_strength # type: ignore[no-any-return]

        # Noise
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        y_0 = pad_sequence(
            [
                torch.randn(
                    d,
                    self.num_channels,
                    device=self.device,
                    dtype=step_cond.dtype,
                    generator=generator
                )
                for d in duration
            ],
            padding_value=0,
            batch_first=True
        )

        t_start = 0.0

        # Inner time step observation
        if duplicate_test:
            t_start = t_inter
            y_0 = (1 - t_start) * y_0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1.0, steps, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        traj = odeint(ode_solver, y_0, t, method=self.ode_solver)

        sample = traj[-1]

        out = sample
        out = torch.where(cond_mask, cond, out)

        if vocoder is not None:
            out = vocoder(out.permute(0, 2, 1))

        return out, traj
