import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from typing_extensions import Literal

from torch.nn.utils import spectral_norm

RESAMPLE_TYPE_LITERAL = Literal["timepreserve", "half"]
RESAMPLE_TYPE = Optional[RESAMPLE_TYPE_LITERAL]

__all__ = [
    "RESAMPLE_TYPE",
    "LearnedDownsample",
    "LearnedUpsample",
    "Downsample",
    "Upsample",
]

class LearnedDownsample(nn.Module):
    """
    Learned downsampling layer
    """
    conv: nn.Module

    def __init__(
        self,
        dim_in: int,
        layer_type: RESAMPLE_TYPE=None,
    ) -> None:
        """
        :param dim_in: input dimension
        :param layer_type: type of downsampling layer
        """
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type is None:
            self.conv = nn.Identity()
        elif self.layer_type == "timepreserve":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 1),
                    stride=(2, 1),
                    groups=dim_in,
                    padding=(1, 0)
                )
            )
        elif self.layer_type == "half":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    groups=dim_in,
                    padding=1
                )
            )
        else:
            raise TypeError(f"Got unexpected donwsample type {layer_type}, expected [timepreserve, half] or None")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Either convolve or return the input tensor
        """
        return self.conv(x) # type: ignore[no-any-return]

class LearnedUpsample(nn.Module):
    """
    Learned upsampling layer
    """
    conv: nn.Module

    def __init__(
        self,
        dim_in: int,
        layer_type: RESAMPLE_TYPE=None,
    ) -> None:
        """
        :param dim_in: input dimension
        :param layer_type: type of upsampling layer
        """
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type is None:
            self.conv = nn.Identity()
        elif self.layer_type == "timepreserve":
            self.conv = nn.ConvTranspose2d(
                dim_in,
                dim_in,
                kernel_size=(3, 1),
                stride=(2, 1),
                groups=dim_in,
                output_padding=(1, 0),
                padding=(1, 0)
            )
        elif self.layer_type == "half":
            self.conv = nn.ConvTranspose2d(
                dim_in,
                dim_in,
                kernel_size=(3, 3),
                stride=(2, 2),
                groups=dim_in,
                output_padding=1,
                padding=1
            )
        else:
            raise TypeError(f"Got unexpected upsample type {layer_type}, expected [timepreserve, half] or None")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Either convolve or return the input tensor
        """
        return self.conv(x) # type: ignore[no-any-return]

class Downsample(nn.Module):
    """
    Downsample layer
    """
    def __init__(self, layer_type: RESAMPLE_TYPE=None) -> None:
        """
        :param layer_type: type of downsampling layer
        """
        super().__init__()
        self.layer_type = layer_type
        if self.layer_type not in ["timepreserve", "half", None]:
            raise TypeError(f"Got unexpected donwsample type {layer_type}, expected [timepreserve, half] or None")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        :return: downsampled tensor or the input tensor
        """
        if self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == "half":
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        return x

class Upsample(nn.Module):
    """
    Upsample layer
    """
    def __init__(self, layer_type: RESAMPLE_TYPE=None) -> None:
        """
        :param layer_type: type of upsampling layer
        """
        super().__init__()
        self.layer_type = layer_type
        if self.layer_type not in ["timepreserve", "half", None]:
            raise TypeError(f"Got unexpected donwsample type {layer_type}, expected [timepreserve, half] or None")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        :return: upsampled tensor or the input tensor
        """
        if self.layer_type == "timepreserve":
            return F.interpolate(x, scale_factor=(2, 1), mode="nearest") # type: ignore[no-any-return]
        elif self.layer_type == "half":
            return F.interpolate(x, scale_factor=2, mode="nearest") # type: ignore[no-any-return]
        return x
