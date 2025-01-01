import torch
import torch.nn as nn

from typing import Optional

from .base import Module

__all__ = ["FeedForward"]

class FeedForward(Module):
    """
    FeedForward module with GELU activation and dropout.
    """
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int]=None,
        mult: int=4,
        dropout: float=0.0,
        approximate: str="none"
    ) -> None:
        """
        :param dim: input dimension
        :param dim_out: output dimension. When None, same as input
        :param mult: multiplier for inner dimension
        :param dropout: dropout probability
        :param approximate: whether to use approximate GELU
        """
        super().__init__()
        inner_dim = int(dim * mult)
        outer_dim = dim_out if dim_out is not None else dim

        gelu = nn.GELU(approximate=approximate)
        linear_in = nn.Linear(dim, inner_dim)
        project_in = nn.Sequential(linear_in, gelu)

        self.ff = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, outer_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, seq_len, dim)
        :return: (batch_size, seq_len, dim_out)
        """
        return self.ff(x) # type: ignore[no-any-return]
