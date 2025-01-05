import torch
import torch.nn as nn

from typing import Tuple

from .layers import (
    DurationEncoder,
    LinearNorm,
    AdaINResBlock1D
)

__all__ = ["ProsodyPredictor"]

class ProsodyPredictor(nn.Module):
    """
    Prosody predictor module
    """
    def __init__(
        self,
        style_dim: int,
        hidden_dim: int,
        num_layers: int,
        max_duration: int=50,
        dropout: float=0.1
    ) -> None:
        """
        :param style_dim: dimension of style embedding
        :param hidden_dim: dimension of hidden states
        :param num_layers: number of layers
        :param max_duration: maximum duration
        :param dropout: dropout rate
        """
        super().__init__() 

        self.text_encoder = DurationEncoder(
            style_dim=style_dim, 
            model_dim=hidden_dim,
            num_layers=num_layers, 
            dropout=dropout
        )
        self.lstm = nn.LSTM(
            hidden_dim + style_dim,
            hidden_dim // 2,
            1,
            batch_first=True,
            bidirectional=True
        )
        self.shared = nn.LSTM(
            hidden_dim + style_dim,
            hidden_dim // 2,
            1,
            batch_first=True,
            bidirectional=True
        )
        self.f0 = nn.ModuleList([
            AdaINResBlock1D(hidden_dim, hidden_dim, style_dim, dropout_p=dropout),
            AdaINResBlock1D(hidden_dim, hidden_dim // 2, style_dim, upsample="half", dropout_p=dropout),
            AdaINResBlock1D(hidden_dim // 2, hidden_dim // 2, style_dim, dropout_p=dropout)
        ])
        self.n = nn.ModuleList([
            AdaINResBlock1D(hidden_dim, hidden_dim, style_dim, dropout_p=dropout),
            AdaINResBlock1D(hidden_dim, hidden_dim // 2, style_dim, upsample="half", dropout_p=dropout),
            AdaINResBlock1D(hidden_dim // 2, hidden_dim // 2, style_dim, dropout_p=dropout),
        ])

        self.duration_proj = LinearNorm(hidden_dim, max_duration)
        self.f0_proj = nn.Conv1d(hidden_dim // 2, 1, 1, 1, 0)
        self.n_proj = nn.Conv1d(hidden_dim // 2, 1, 1, 1, 0)

    def forward(
        self,
        texts: torch.Tensor,
        style: torch.Tensor,
        text_lengths: torch.Tensor,
        alignment: torch.Tensor,
        m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param texts: text tensor [batch, seq]
        :param style: style tensor [batch, style_dim]
        :param text_lengths: text lengths tensor [batch]
        :param alignment: alignment tensor [batch, seq, seq]
        :param m: mask tensor [batch, seq]
        :return: duration tensor [batch, seq], energy tensor [batch, seq, seq]
        """
        d = self.text_encoder(texts, style, text_lengths, m)
        batch_size = d.shape[0]
        text_size = d.shape[1]

        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d,
            input_lengths, # type: ignore[arg-type]
            batch_first=True,
            enforce_sorted=False
        )

        m = m.to(text_lengths.device).unsqueeze(1)
        self.lstm.flatten_parameters()

        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence( # type: ignore[assignment]
            x,
            batch_first=True
        )

        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]]) # type: ignore[attr-defined]
        x_pad[:, :x.shape[1], :] = x # type: ignore[attr-defined,assignment]
        x = x_pad.to(x.device) # type: ignore[attr-defined,assignment]

        duration = self.duration_proj(
            nn.functional.dropout(
                x, # type: ignore[arg-type]
                0.5,
                training=self.training
            )
        )

        en = (d.transpose(-1, -2) @ alignment)
        return duration.squeeze(-1), en

    def f0_n_train(
        self,
        x: torch.Tensor,
        s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: input tensor [batch, seq, hidden_dim]
        :param s: style tensor [batch, style_dim]
        :return: f0 tensor [batch, seq], n tensor [batch, seq]
        """
        x, _ = self.shared(x.transpose(-1, -2))

        f0 = x.transpose(-1, -2)
        for block in self.f0:
            f0 = block(f0, s)

        f0 = self.f0_proj(f0)
        n = x.transpose(-1, -2)
        for block in self.n:
            n = block(n, s)

        n = self.n_proj(n)
        return f0.squeeze(1), n.squeeze(1)
