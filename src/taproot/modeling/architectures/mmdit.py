import torch
import torch.nn as nn

from typing import Optional
from torch import Tensor

from ..modules import (
    Module,
    AdaptiveFinalLayerNorm,
)
from ..embeddings import (
    TimestepEmbedding,
    TextEmbedding,
    AudioEmbedding,
    RotaryEmbedding
)
from ..blocks import MMDiTBlock

__all__ = ["MMDiT"]

class MMDiT(Module):
    """
    Masked Multimodal Diffusion Transformer (MMDiT) model.

    Proposed by Esser et. al. in "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (2024).

    :see https://arxiv.org/abs/2403.03206:
    :see https://github.com/SWivid/F5-TTS/blob/main/model/backbones/mmdit.py:
    """
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
    ) -> None:
        super().__init__()
        self.time_embed = TimestepEmbedding(dim=dim)
        self.text_embed = TextEmbedding(out_dim=dim, text_num_embeds=text_num_embeds)
        self.rotary_embed = RotaryEmbedding(dim=dim_head)
        self.audio_embed = AudioEmbedding(in_dim=mel_dim, out_dim=dim)
        self.transformer_blocks = nn.ModuleList([
            MMDiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                ff_mult=ff_mult,
                context_pre_only=i == depth-1,
            ) for i in range(depth)
        ])
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
        c = self.text_embed(text, drop_text=drop_text)
        x = self.audio_embed(x, cond, drop_audio_cond=drop_audio_cond)

        rope_audio = self.rotary_embed(torch.arange(seq_len, device=x.device))
        rope_text = self.rotary_embed(torch.arange(text_len, device=x.device))

        for block in self.transformer_blocks:
            c, x = block(x, c, t, mask=mask, rope=rope_audio, c_rope=rope_text)

        x = self.norm_out(x)
        x = self.proj_out(x)

        return x
