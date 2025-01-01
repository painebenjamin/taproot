from __future__ import annotations
# adapted from https://github.com/hzwer/Practical-RIFE/
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, Tuple

__all__ = ["Warper"]

class Warper(nn.Module):
    """
    Warper module that warps an image using a flow field.
    """
    backwarp_grid_cache: Dict[Tuple[int, ...], torch.Tensor]

    def __init__(self) -> None:
        super(Warper, self).__init__()
        self.backwarp_grid_cache = {}

    def to(self, *args: Any, **kwargs: Any) -> Warper:
        """
        Moves and/or casts the parameters of the warper to the specified device and dtype.
        """
        super().to(*args, **kwargs)
        for k, v in self.backwarp_grid_cache.items():
            self.backwarp_grid_cache[k] = v.to(*args, **kwargs)
        return self

    def forward(
        self,
        input_tensor: torch.Tensor,
        flow_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Warps the input tensor using the flow tensor.
        """
        grid_key = flow_tensor.size()
        if grid_key not in self.backwarp_grid_cache:
            horizontal_tensor = torch.linspace(-1.0, 1.0, flow_tensor.shape[3])
            horizontal_tensor = horizontal_tensor.view(1, 1, 1, flow_tensor.shape[3])
            horizontal_tensor = horizontal_tensor.expand(flow_tensor.shape[0], -1, flow_tensor.shape[2], -1)

            vertical_tensor = torch.linspace(-1.0, 1.0, flow_tensor.shape[2])
            vertical_tensor = vertical_tensor.view(1, 1, flow_tensor.shape[2], 1)
            vertical_tensor = vertical_tensor.expand(flow_tensor.shape[0], -1, -1, flow_tensor.shape[3])
            self.backwarp_grid_cache[grid_key] = torch.cat([
                horizontal_tensor,
                vertical_tensor
            ], dim=1).to(input_tensor.device, dtype=input_tensor.dtype)

        flow_tensor = torch.cat([
            flow_tensor[:, 0:1, :, :] / ((input_tensor.shape[3] - 1.0) / 2.0),
            flow_tensor[:, 1:2, :, :] / ((input_tensor.shape[2] - 1.0) / 2.0)
        ], dim=1)
        grid = (self.backwarp_grid_cache[grid_key] + flow_tensor).permute(0, 2, 3, 1)

        return F.grid_sample(
            input=input_tensor,
            grid=grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
