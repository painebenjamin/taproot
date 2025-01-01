import torch
import torch.nn as nn

from torch import Tensor

from ..modules import Module
from .position import ConvolutionalPositionEmbedding

__all__ = [
    "AudioEmbedding",
    "AudioTextEmbedding"
]

class AudioEmbedding(Module):
    """
    Audio embedding module
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        """
        :param in_dim: input dimension
        :param out_dim: output dimension
        """
        super().__init__()
        self.proj = nn.Linear(2 * in_dim, out_dim)
        self.conv_pos_embed = ConvolutionalPositionEmbedding(out_dim)

    def forward(
        self,
        x: Tensor, # (B, T, C)
        cond: Tensor, # (B, T, C)
        drop_audio_cond: bool=False
    ) -> Tensor:
        """
        :param x: input tensor
        :param cond: condition tensor
        :param drop_audio_cond: whether to drop the condition tensor
        :return: output tensor
        """
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = torch.cat([x, cond], dim=-1)
        x = self.proj(x)
        x = self.conv_pos_embed(x) + x
        return x

class AudioTextEmbedding(Module):
    """
    Audio embedding module with text
    """
    def __init__(
        self,
        in_dim: int,
        text_dim: int,
        out_dim: int
    ) -> None:
        """
        :param in_dim: input dimension
        :param text_dim: text dimension
        :param out_dim: output dimension
        """
        super().__init__()
        self.proj = nn.Linear(2 * in_dim + text_dim, out_dim)
        self.conv_pos_embed = ConvolutionalPositionEmbedding(out_dim)

    def forward(
        self,
        x: Tensor, # (B, T, C)
        cond: Tensor, # (B, T, C)
        text: Tensor, # (B, T, C)
        drop_audio_cond: bool=False
    ) -> Tensor:
        """
        :param x: input tensor
        :param cond: condition tensor
        :param drop_audio_cond: whether to drop the condition tensor
        :return: output tensor
        """
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = torch.cat([x, cond, text], dim=-1)
        x = self.proj(x)
        x = self.conv_pos_embed(x) + x
        return x
