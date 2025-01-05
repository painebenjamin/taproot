import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from torch.nn.utils import spectral_norm, weight_norm

from .resample import Downsample, LearnedDownsample, RESAMPLE_TYPE

__all__ = ["ResBlock", "ResBlock1D"]

class ResBlock(nn.Module):
    """
    Residual block with spectral normalization.
    """
    norm1: nn.Module
    norm2: nn.Module
    conv1x1: nn.Module

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        lrelu_slope: float=0.2,
        normalize: bool=False,
        downsample: RESAMPLE_TYPE=None
    ) -> None:
        """
        :param in_dim: Number of input channels.
        :param out_dim: Number of output channels.
        :param lrelu_slope: Slope of the negative part of the LeakyReLU activation function.
        :param normalize: Whether to apply instance normalization.
        :param downsample: Type of downsampling to apply.
        """
        super().__init__()
        self.actv = nn.LeakyReLU(lrelu_slope)

        self.downsample = Downsample(downsample)
        self.downsample_res = LearnedDownsample(in_dim, downsample)

        self.conv1 = spectral_norm(nn.Conv2d(in_dim, in_dim, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(in_dim, out_dim, 3, 1, 1))

        if normalize:
            self.norm1 = nn.InstanceNorm2d(in_dim, affine=True)
            self.norm2 = nn.InstanceNorm2d(in_dim, affine=True)
        else:
            self.norm1 = self.norm2 = nn.Identity()

        if in_dim != out_dim:
            self.conv1x1 = spectral_norm(nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False))
        else:
            self.conv1x1 = nn.Identity()

    def shortcut(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shortcut connection.
        """
        x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Residual connection.
        """
        x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        x = self.shortcut(x) + self.residual(x)
        return x / sqrt(2)  # unit variance

class ResBlock1D(nn.Module):
    """
    Residual block for 1D signals.
    """
    pool: nn.Module
    norm1: nn.Module
    norm2: nn.Module
    conv1x1: nn.Module

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        lrelu_slope: float=0.2,
        normalize: bool=False,
        downsample: RESAMPLE_TYPE=None,
        dropout_p: float=0.2
    ) -> None:
        """
        :param in_dim: Number of input channels.
        :param out_dim: Number of output channels.
        :param lrelu_slope: Slope of the negative part of the LeakyReLU activation function.
        :param normalize: Whether to apply instance normalization.
        :param downsample: Type of downsampling to apply.
        :param dropout_p: Dropout probability.
        """
        super().__init__()
        self.actv = nn.LeakyReLU(lrelu_slope)
        self.downsample_type = downsample
        self.learned_sc = in_dim != out_dim
        self.dropout_p = dropout_p

        if self.downsample_type is None:
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.Conv1d(
                    in_dim,
                    in_dim,
                    kernel_size=3,
                    stride=2,
                    groups=in_dim,
                    padding=1
                )
            )

        self.conv1 = weight_norm(nn.Conv1d(in_dim, in_dim, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(in_dim, out_dim, 3, 1, 1))

        if normalize:
            self.norm1 = nn.InstanceNorm1d(in_dim, affine=True)
            self.norm2 = nn.InstanceNorm1d(in_dim, affine=True)
        else:
            self.norm1 = self.norm2 = nn.Identity()

        if in_dim != out_dim:
            self.conv1x1 = weight_norm(nn.Conv1d(in_dim, out_dim, 1, 1, 0, bias=False))
        else:
            self.conv1x1 = nn.Identity()

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample input tensor.
        """
        if self.downsample_type is None:
            return x
        if x.shape[-1] % 2 != 0:
            x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
        return F.avg_pool1d(x, 2)

    def shortcut(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shortcut connection.
        """
        x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Residual connection.
        """
        x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm2(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        x = self.shortcut(x) + self.residual(x)
        return x / sqrt(2)  # unit variance
