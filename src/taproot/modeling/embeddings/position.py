import torch
import math

import torch.nn as nn

from typing import Optional
from torch import Tensor

__all__ = [
    "SinusoidalPositionEmbedding",
    "ConvolutionalPositionEmbedding",
]

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for transformers.
    """
    def __init__(self, dim: int) -> None:
        """
        :param dim: int, dimension of the input tensor
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor, scale: float=1000.0) -> Tensor:
        """
        :param x: Tensor, shape [seq_len, dim]
        :param scale: float, scale of the sinusoidal encoding
        :return: Tensor, shape [seq_len, dim]
        """
        half_dim = self.dim // 2
        e = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -e)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConvolutionalPositionEmbedding(nn.Module):
    """
    Convolutional positional encoding for transformers.
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int=31,
        groups: int=16
    ) -> None:
        """
        :param dim: int, dimension of the input tensor
        :param kernel_size: int, size of the convolutional kernel
        :param groups: int, number of groups in the convolutional layer
        """
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(
        self,
        x: Tensor, # shape: [b n d]
        mask: Optional[Tensor]=None # shape: [b n]
    ) -> Tensor:
        """
        :param x: Tensor, shape [b n d]
        :param mask: Optional tensor, shape [b n]
        :return: Tensor, shape [b n d]
        """
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out
