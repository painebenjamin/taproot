import torch
import torch.nn as nn

from typing import Tuple
from torch import Tensor
from .base import Module

__all__ = [
    "GlobalResponseNorm",
    "AdaptiveLayerNorm",
    "AdaptiveFinalLayerNorm"
]

class GlobalResponseNorm(Module):
    """
    Global Response Normalization

    Output follows the formula:
        y = gamma * (x * n_x) + beta + x
    where:
        n_x = x / sqrt(mean(x^2) + eps)
    """
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6
    ) -> None:
        """
        :param dim: dimension of input feature
        :param eps: epsilon value for numerical stability
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: input feature
        :return: normalized feature
        """
        g_x = torch.norm(x, p=2, dim=1, keepdim=True)
        n_x = g_x / (g_x.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * n_x) + self.beta + x # type: ignore[no-any-return]

class AdaptiveLayerNorm(Module):
    """
    Adaptive Layer Normalization
    """
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6
    ) -> None:
        """
        :param dim: dimension of input feature
        :param eps: epsilon value for numerical stability
        """
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(
        self,
        x: Tensor,
        emb: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        :param x: input feature
        :param emb: embedding feature
        :return: normalized features
        """
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class AdaptiveFinalLayerNorm(Module):
    """
    Adaptive Final Layer Normalization
    """
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6
    ) -> None:
        """
        :param dim: dimension of input feature
        :param eps: epsilon value for numerical stability
        """
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        :param x: input feature
        :param emb: embedding feature
        :return: normalized features
        """
        emb = self.linear(self.silu(emb))
        scale, shift = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
