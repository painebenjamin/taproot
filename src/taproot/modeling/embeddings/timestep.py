import torch
import torch.nn as nn

from ..modules import Module
from .position import SinusoidalPositionEmbedding

__all__ = ["TimestepEmbedding"]

class TimestepEmbedding(Module):
    """
    Timestep embedding module
    """
    def __init__(
        self,
        dim: int,
        freq_embed_dim: int=256
    ) -> None:
        """
        :param dim: dimension of the input tensor
        :param freq_embed_dim: dimension of the frequency embedding
        """
        super().__init__()
        self.time_embed = SinusoidalPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        :return: output tensor
        """
        x = self.time_embed(x).to(dtype=x.dtype)
        x = self.time_mlp(x) # b, d
        return x
