import torch
import torch.nn as nn

from typing import Optional
from torch import Tensor

from taproot.util import (
    precompute_freqs_cis,
    get_pos_embed_indices
)

from ..blocks import ConvNeXtV2Block
from ..modules import Module

__all__ = [
    "TextEmbedding",
    "ConvolutionalTextEmbedding",
]

class TextEmbedding(Module):
    """
    Text embedding module that adds positional embeddings to text tokens.
    """
    def __init__(
        self,
        out_dim: int,
        text_num_embeds: int,
        precompute_max_pos: int=1024,
    ) -> None:
        """
        :param out_dim: int dimension of output embeddings
        :param text_num_embeds: int number of text embeddings
        :param precompute_max_pos: int maximum number of positions to precompute
        """
        super().__init__()
        # Add one for filler token
        self.text_embed = nn.Embedding(text_num_embeds + 1, out_dim)
        self.precompute_max_pos = precompute_max_pos
        # Precompute positional embeddings
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(out_dim, self.precompute_max_pos),
            persistent=False,
        )

    def forward(
        self,
        text: Tensor,
        seq_len: Optional[int]=None,
        drop_text: bool=False
    ) -> Tensor:
        """
        :param text: (B, T) tensor of text tokens
        :param seq_len: int length of sequence, optional
        :param drop_text: bool indicating whether to drop text embeddings
        :return: (B, T, D) tensor of text embeddings
        """
        text = text + 1
        if seq_len is not None:
            text = text[:, :seq_len]
            text = torch.nn.functional.pad(
                text,
                (0, seq_len - text.size(1)),
                mode="constant",
                value=0,
            )

        if drop_text:
            text = torch.zeros_like(text)

        text = self.text_embed(text)
        b, t, d = text.shape

        batch_start = torch.zeros((b,), dtype=torch.long, device=text.device)
        pos_idx = get_pos_embed_indices(
            batch_start,
            t,
            max_pos=self.precompute_max_pos,
        )
        text_pos_embed = self.freqs_cis[pos_idx]
        text = text + text_pos_embed
        return text

class ConvolutionalTextEmbedding(TextEmbedding):
    """
    Text embedding module that adds positional embeddings to text tokens.

    This module uses a convolutional layer to learn text embeddings.
    """
    def __init__(
        self,
        out_dim: int,
        text_num_embeds: int,
        precompute_max_pos: int=4096,
        conv_layers: int=4,
        conv_mult: int=2,
    ) -> None:
        """
        :param out_dim: int dimension of output embeddings
        :param text_num_embeds: int number of text embeddings
        :param precompute_max_pos: int maximum number of positions to precompute
        :param conv_layers: int number of convolutional layers
        :param conv_mult: int convolutional layer multiplier
        """
        super().__init__(
            out_dim=out_dim,
            text_num_embeds=text_num_embeds,
            precompute_max_pos=precompute_max_pos
        )
        self.text_blocks = nn.Sequential(*[
            ConvNeXtV2Block(
                dim=out_dim,
                intermediate_dim=out_dim * conv_mult,
            )
            for i in range(conv_layers)
        ])

    def forward(
        self,
        text: Tensor,
        seq_len: Optional[int]=None,
        drop_text: bool=False
    ) -> Tensor:
        """
        :param text: (B, T) tensor of text tokens
        :param seq_len: int length of sequence, optional
        :param drop_text: bool indicating whether to drop text embeddings
        :return: (B, T, D) tensor of text embeddings
        """
        text = super().forward(text, seq_len=seq_len, drop_text=drop_text)
        text = self.text_blocks(text)
        return text
