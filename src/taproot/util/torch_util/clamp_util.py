from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["finite_clamp", "float16_finite_clamp"]

def finite_clamp(x: torch.Tensor, offset: float = 1e4) -> torch.Tensor:
    """
    Clamps a tensor if it contains infinite values.

    Uses `torch.finfo` to get the maximum value of the tensor's dtype and
    subtracts the offset to get the clamp value.
    """
    import torch
    clamp_value = torch.finfo(x.dtype).max - offset
    return torch.clamp(x, -clamp_value, clamp_value)

def float16_finite_clamp(x: torch.Tensor, offset: float = 1e4) -> torch.Tensor:
    """
    Clamps a tensor if it is float16 and contains infinite values.

    Uses `torch.finfo` to get the maximum value of the tensor's dtype and
    subtracts the offset to get the clamp value.
    """
    import torch
    if x.dtype is torch.float16:
        return finite_clamp(x, offset)
    return x
