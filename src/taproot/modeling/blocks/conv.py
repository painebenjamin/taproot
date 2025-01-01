import torch
import torch.nn as nn

from ..modules import GlobalResponseNorm

class ConvNeXtV2Block(nn.Module):
    """
    ConvNeXtV2 block, as described in the paper:

        ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
        Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie

    :see https://arxiv.org/abs/2301.00808:
    :see https://github.com/SWivid/F5-TTS/blob/main/model/modules.py:
    :see https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py:
    """
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int=1,
        eps: float=1e-6
    ):
        """
        :param dim: input/output dimension
        :param intermediate_dim: intermediate dimension for the pointwise convolutions
        :param dilation: dilation factor
        """
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            padding=padding,
            groups=dim,
            dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(intermediate_dim) # global response normalization
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        :return: output tensor
        """
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x
