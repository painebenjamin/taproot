import torch
import torch.nn as nn

from typing import Optional, Union
from torch import Tensor

from ..modules import (
    Module,
    AdaptiveFinalLayerNorm,
)
from ..embeddings import (
    ConvolutionalTextEmbedding,
    TimestepEmbedding,
    TextEmbedding,
    AudioTextEmbedding,
    RotaryEmbedding
)
from ..blocks import DiTBlock

__all__ = ["DiT"]

class DiT(Module):
    """
    Diffusion Transformer (DiT) model.

    :see https://github.com/SWivid/F5-TTS/blob/main/model/backbones/dit.py:
    """
    text_embed: Union[ConvolutionalTextEmbedding, TextEmbedding]
    long_skip_connection: Optional[nn.Linear]

    def __init__(
        self,
        dim: int,
        depth: int=8,
        heads: int=8,
        dim_head: int=64,
        dropout: float=0.1,
        ff_mult: int=4,
        text_num_embeds: int=256,
        mel_dim: int=100,
        text_dim: Optional[int]=None,
        conv_layers: Optional[int]=None,
        long_skip_connection: bool=False,
    ) -> None:
        """
        :param dim: model dimension
        :param depth: number of transformer blocks
        :param heads: number of attention heads
        :param dim_head: dimension of each attention head
        :param dropout: dropout rate
        :param ff_mult: feedforward multiplier
        :param text_num_embeds: number of text embeddings
        :param mel_dim: mel dimension
        :param text_dim: text dimension
        :param conv_layers: number of convolutional layers
        :param long_skip_connection: whether to use long skip connection
        """
        super().__init__()
        self.dim = dim
        cond_dim = text_dim if text_dim is not None else dim

        if conv_layers is not None and conv_layers > 0:
            self.text_embed = ConvolutionalTextEmbedding(
                out_dim=cond_dim,
                text_num_embeds=text_num_embeds,
                conv_layers=conv_layers,
            )
        else:
            self.text_embed = TextEmbedding(
                out_dim=cond_dim,
                text_num_embeds=text_num_embeds
            )

        self.time_embed = TimestepEmbedding(dim=dim)
        self.rotary_embed = RotaryEmbedding(dim=dim_head)
        self.input_embed = AudioTextEmbedding(
            in_dim=mel_dim,
            text_dim=cond_dim,
            out_dim=dim
        )
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                ff_mult=ff_mult,
            ) for i in range(depth)
        ])

        if long_skip_connection:
            self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False)
        else:
            self.long_skip_connection = None

        self.norm_out = AdaptiveFinalLayerNorm(dim=dim)
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
        self,
        x: Tensor, # noised audio, shape (batch, seq, dim)
        cond: Tensor, # masked cond audio (batch, seq, dim)
        text: Tensor, # text tokens, shape (batch, text_seq)
        time: Tensor, # timestep, shape (batch,)
        drop_audio_cond: bool=False,
        drop_text: bool=False,
        mask: Optional[Tensor]=None, # optional mask, (batch, seq)
    ) -> Tensor:
        """
        :param x: noised audio, shape (batch, seq, dim)
        :param cond: masked cond audio (batch, seq, dim)
        :param text: text tokens, shape (batch, text_seq)
        :param time: timestep, shape (batch,)
        :param drop_audio_cond: bool, whether to drop audio condition
        :param drop_text: bool, whether to drop text
        :param mask: optional mask, (batch, seq)
        """
        batch, seq_len = x.shape[:2]
        text_len = text.shape[1]

        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        c = self.text_embed(text, seq_len=seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, c, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed(torch.arange(seq_len, device=x.device))

        if self.long_skip_connection is not None:
            res = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat([x, res], dim=-1))

        x = self.norm_out(x, t)
        x = self.proj_out(x)

        return x
