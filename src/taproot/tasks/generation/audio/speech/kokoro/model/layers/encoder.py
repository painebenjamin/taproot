import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn.utils import weight_norm

from .ada import AdaLN
from .normalize import LayerNorm

__all__ = [
    "TextEncoder",
    "DurationEncoder"
]

class TextEncoder(nn.Module):
    """
    Text Encoder
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        num_layers: int,
        num_symbols: int,
        lrelu_slope: float=0.2,
        dropout: float=0.2,
    ) -> None:
        """
        :param channels: the number of channels
        :param kernel_size: the size of kernel
        :param num_layers: the number of layers
        :param num_symbols: the number of symbols
        :param lrelu_slope: the slope of leaky relu
        :param dropout: the dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(num_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                nn.LeakyReLU(lrelu_slope),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.lstm = nn.LSTM(
            channels,
            channels//2,
            1,
            batch_first=True,
            bidirectional=True
        )

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the input tensor [B, T]
        :return: the output tensor [B, T, chn]
        """
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        m: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: the input tensor [B, T]
        :param input_lengths: the lengths of input tensor [B]
        :param m: the mask tensor [B, T]
        :return: the output tensor [B, T, chn]
        """
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)

        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)

        x = x.transpose(1, 2)  # [B, T, chn]
        x = nn.utils.rnn.pack_padded_sequence( # type: ignore[assignment]
            x,
            input_lengths.cpu().numpy(), # type: ignore[arg-type]
            batch_first=True,
            enforce_sorted=False
        )

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, # type: ignore[arg-type]
            batch_first=True
        )

        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device, x.dtype)
        
        x.masked_fill_(m, 0.0)
        return x

class DurationEncoder(nn.Module):
    """
    Duration Encoder
    """
    def __init__(
        self,
        style_dim: int,
        model_dim: int,
        num_layers: int,
        dropout: float=0.1
    ) -> None:
        """
        :param style_dim: the dimension of style
        :param model_dim: the dimension of model
        :param num_layers: the number of layers
        :param dropout: the dropout rate
        """
        super().__init__()
        self.dropout = dropout
        self.model_dim = model_dim
        self.style_dim = style_dim
        self.lstms = nn.ModuleList()
        for _ in range(num_layers):
            self.lstms.append(
                nn.LSTM(
                    model_dim + style_dim, 
                    model_dim // 2, 
                    num_layers=1, 
                    batch_first=True, 
                    bidirectional=True, 
                    dropout=dropout
                )
            )
            self.lstms.append(
                AdaLN(
                    style_dim,
                    model_dim
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        text_lengths: torch.Tensor,
        m: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: the input tensor [B, T, chn]
        :param style: the style tensor [B, style]
        :param text_lengths: the lengths of text tensor [B]
        :param m: the mask tensor [B, T]
        :return: the output tensor [B, T, chn]
        """
        masks = m.to(text_lengths.device)

        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], dim=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)

        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)

        for block in self.lstms:
            if isinstance(block, AdaLN):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], dim=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence( # type: ignore[assignment]
                    x,
                    input_lengths, # type: ignore[arg-type]
                    batch_first=True,
                    enforce_sorted=False
                )
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, # type: ignore[arg-type]
                    batch_first=True
                )
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)

                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad.to(x.device, x.dtype)

        return x.transpose(-1, -2)

    def inference(
        self,
        x: torch.Tensor,
        style: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: the input tensor [B, T, chn]
        :param style: the style tensor [B, style]
        :return: the output tensor [B, T, chn]
        """
        x = self.embedding(x.transpose(-1, -2)) * np.sqrt(self.model_dim) # type: ignore[operator]
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], dim=-1)
        src = self.pos_encoder(x) # type: ignore[operator]
        output = self.transformer_encoder(src).transpose(0, 1) # type: ignore[operator]
        return output # type: ignore[no-any-return]
