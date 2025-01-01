import torch

from typing import Tuple
from torch import Tensor
from einops import rearrange

from ..modules import Module

__all__ = ["RotaryEmbedding"]

class RotaryEmbedding(Module):
    """
    Rotary embedding module.

    :see https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py:
    """
    def __init__(
        self,
        dim: int,
        use_xpos: bool=False,
        scale_base: int=512,
        interpolation_factor: float=1.0,
        base: int=10_000,
        base_rescale_factor: float=1.0
    ) -> None:
        """
        :param dim: input dimension
        :param use_xpos: whether to use positional encoding
        :param scale_base: scale base
        :param interpolation_factor: interpolation factor
        :param base: base
        :param base_rescale_factor: base rescale factor
        """
        super().__init__()

        assert interpolation_factor >= 1.0, "interpolation factor must be >= 1"

        base = base * base_rescale_factor ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        self.scale_base = scale_base
        self.register_buffer("inv_freq", inv_freq)
        self.interpolation_factor = interpolation_factor

        if use_xpos:
            scale = (torch.arange(0, dim, 2) + 0.4 * dim).unsqueeze(0) / (1.4 * dim)
            self.register_buffer("scale", scale)
        else:
            self.register_buffer("scale", None)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        max_pos = x.max() + 1
        freqs = torch.einsum("i , j -> i j", x.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = torch.stack((freqs, freqs), dim=-1)
        freqs = rearrange(freqs, "... d r -> ... (d r)")

        if self.scale is None:
            return freqs, torch.tensor(1.0)

        power = (x - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.stack((scale, scale), dim=-1)
        scale = rearrange(scale, "... d r -> ... (d r)")

        return freqs, scale
