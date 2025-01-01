import torch.nn as nn

from typing import Optional, Tuple
from torch import Tensor

from ..modules import (
    Module,
    FeedForward,
    AdaptiveLayerNorm,
    AdaptiveFinalLayerNorm,
    Attention,
    JointAttention
)

__all__ = ["DiTBlock", "MMDiTBlock"]

class DiTBlock(Module):
    """
    Diffusion Transformer Block
    """
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: int=4,
        dropout: float=0.1
    ) -> None:
        """
        :param dim: dimension of input tensor
        :param heads: number of heads for multi-head attention
        :param dim_head: dimension of each head
        :param ff_mult: multiplier for feed-forward network
        :param dropout: dropout rate
        """
        super().__init__()

        self.attn_norm = AdaptiveLayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(
        self,
        x: Tensor, # Noised input
        t: Tensor, # Time embedding
        mask: Optional[Tensor]=None,
        rope: Optional[Tensor]=None
    ) -> Tensor:
        """
        :param x: input tensor
        :param t: time embedding
        :param mask: attention mask
        :param rope: relative positional encoding
        :return: output tensor
        """
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x

class MMDiTBlock(nn.Module):
    """
    Multi-Modal Diffusion Transformer Block
    :see https://arxiv.org/abs/2403.03206:
    """
    ff_c: Optional[FeedForward]
    ff_norm_c: Optional[nn.LayerNorm]

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: int=4,
        dropout: float=0.1,
        context_pre_only: Optional[bool]=False,
    ) -> None:
        """
        :param dim: dimension of input tensor
        :param heads: number of heads for multi-head attention
        :param dim_head: dimension of each head
        :param ff_mult: multiplier for feed-forward network
        :param dropout: dropout rate
        :param context_pre_only: whether to apply context pre-norm only
        """
        super().__init__()

        self.context_pre_only = context_pre_only

        self.attn_norm_c = AdaptiveFinalLayerNorm(dim) if context_pre_only else AdaptiveLayerNorm(dim)
        self.attn_norm_x = AdaptiveLayerNorm(dim)
        self.attn = JointAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=dim,
            context_pre_only=context_pre_only,
        )

        if not context_pre_only:
            self.ff_norm_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_c = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")
        else:
            self.ff_norm_c = None
            self.ff_c = None

        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(
        self,
        x: Tensor, # Noised input
        c: Tensor, # Context
        t: Tensor, # Time embedding
        mask: Optional[Tensor]=None,
        rope: Optional[Tensor]=None,
        c_rope: Optional[Tensor]=None
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        :param x: input tensor
        :param c: context tensor
        :param t: time embedding
        :param mask: attention mask
        :param rope: relative positional encoding for input x
        :param c_rope: relative positional encoding for context c
        :return: output tensor for context c and input x
        """
        # pre-norm & modulation for attention input
        if self.context_pre_only:
            norm_c = self.attn_norm_c(c, t)
        else:
            norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)

        # attention normalization for input x
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)

        # attention
        x_attn_output, c_attn_output = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope)

        # process attention output for context c
        c_out: Optional[Tensor] = None
        if not self.context_pre_only:
            c = c + c_gate_msa.unsqueeze(1) * c_attn_output

            norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None] # type: ignore[misc]
            c_ff_output = self.ff_c(norm_c) # type: ignore[misc]
            c_out = c + c_gate_mlp.unsqueeze(1) * c_ff_output

        # process attention output for input x
        x = x + x_gate_msa.unsqueeze(1) * x_attn_output

        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x_ff_output = self.ff_x(norm_x)
        x = x + x_gate_mlp.unsqueeze(1) * x_ff_output

        return c, x
