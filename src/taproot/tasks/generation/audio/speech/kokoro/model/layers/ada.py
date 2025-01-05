# First adapted from https://github.com/yl4579/StyleTTS2/blob/main/Modules/istftnet.py and https://github.com/yl4579/StyleTTS2/blob/main/Modules/utils.py
# Second adapted from https://huggingface.co/hexgrad/Kokoro-82M/raw/main/istftnet.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from ..util import init_weights, get_padding
from .resample import Upsample, RESAMPLE_TYPE

__all__ = [
    "AdaLN",
    "AdaIN1D",
    "AdaINResBlock",
    "AdaINResBlock1D",
]

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) layer.
    """
    def __init__(
        self,
        style_dim: int,
        num_features: int,
        eps: float=1e-5
    ) -> None:
        """
        :param style_dim: The dimension of the style input.
        :param num_features: The number of features in the input.
        :param eps: The epsilon value for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        :param x: The input tensor.
        :param s: The style tensor.
        :return: The normalized tensor.
        """
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)

        x = F.layer_norm(x, (self.num_features,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)

class AdaIN1D(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) layer for 1D inputs.
    """
    def __init__(self, style_dim: int, num_features: int) -> None:
        """
        :param style_dim: The dimension of the style input.
        :param num_features: The number of features in the input.
        """
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        :param x: The input tensor.
        :param s: The style tensor.
        :return: The normalized tensor.
        """
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta # type: ignore[no-any-return]

class AdaINResBlock(torch.nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) residual block.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int=3,
        dilation: Tuple[int, ...]=(1, 3, 5),
        style_dim: int=64
    ) -> None:
        """
        :param channels: The number of input and output channels.
        :param kernel_size: The size of the kernel.
        :param dilation: The dilation factors of the kernel.
        :param style_dim: The dimension of the style input.
        """
        super(AdaINResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilate,
                    padding=get_padding(kernel_size, dilate)
                )
            )
            for dilate in dilation
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1)
                )
            )
            for _ in dilation
        ])
        self.convs2.apply(init_weights)
        
        self.adain1 = nn.ModuleList([
            AdaIN1D(style_dim, channels)
            for _ in dilation
        ])
        self.adain2 = nn.ModuleList([
            AdaIN1D(style_dim, channels)
            for _ in dilation
        ])

        self.alpha1 = nn.ParameterList([
            nn.Parameter(torch.ones(1, channels, 1))
            for i in range(len(self.convs1))
        ])
        self.alpha2 = nn.ParameterList([
            nn.Parameter(torch.ones(1, channels, 1))
            for i in range(len(self.convs2))
        ])

    def remove_weight_norm(self) -> None:
        """
        Remove weight normalization from the convolutional layers.
        """
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        :param x: The input tensor.
        :param s: The style tensor.
        :return: The output tensor.
        """
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1,
            self.convs2,
            self.adain1,
            self.adain2,
            self.alpha1,
            self.alpha2
        ):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x

        return x

class AdaINResBlock1D(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) residual block for 1D inputs.
    """
    pool: nn.Module
    conv1x1: nn.Module

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        style_dim: int=64,
        upsample: RESAMPLE_TYPE=None,
        dropout_p: float=0.0,
        lrelu_slope: float=0.2
    ) -> None:
        """
        :param dim_in: The number of input channels.
        :param dim_out: The number of output channels.
        :param style_dim: The dimension of the style input.
        :param upsample: The type of upsampling to use.
        :param dropout_p: The dropout probability.
        :param lrelu_slope: The slope for the leaky ReLU activation.
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.actv = nn.LeakyReLU(lrelu_slope)
        self.upsample_type = upsample
        self.upsample = Upsample(upsample)
        self.dropout = nn.Dropout(dropout_p)

        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1D(style_dim, dim_in)
        self.norm2 = AdaIN1D(style_dim, dim_out)

        if upsample is None:
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    groups=dim_in,
                    padding=1,
                    output_padding=1
                )
            )

        if self.learned_sc:
            self.conv1x1 = weight_norm(
                nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)
            )
        else:
            self.conv1x1 = nn.Identity()

    @property
    def learned_sc(self) -> bool:
        """
        :return: Whether the shortcut is learned.
        """
        return self.dim_in != self.dim_out

    def shortcut(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shortcut connection.

        :param x: The input tensor.
        :return: The output tensor.
        """
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def residual(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Residual connection.

        :param x: The input tensor.
        :param s: The style tensor.
        :return: The output tensor.
        """
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: The input tensor.
        :param s: The style tensor.
        :return: The output tensor.
        """
        out = self.residual(x, s)
        out = (out + self.shortcut(x)) / np.sqrt(2)
        return out
