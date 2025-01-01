import torch
import torch.nn as nn
import torch.nn.functional as F

if not hasattr(F, "scaled_dot_product_attention"):
    raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

from typing import Optional, Tuple, Union
from torch import Tensor

from taproot.util import apply_rotary_pos_emb
from .base import Module

__all__ = ["Attention", "JointAttention"]

class Attention(Module):
    """
    Attention module

    :see https://github.com/SWivid/F5-TTS/blob/main/model/modules.py:
    :see https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py:
    """
    def __init__(
        self,
        dim: int,
        heads: int=8,
        dim_head: int=64,
        dropout: float=0.0,
        context_dim: Optional[int]=None,  # if not None -> joint attention
        context_pre_only: Optional[bool]=None
    ) -> None:
        """
        :param dim: input dimension
        :param heads: number of heads
        :param dim_head: dimension of each head
        :param dropout: dropout rate
        :param context_dim: context dimension. If not None, apply joint attention
        :param context_pre_only: whether to apply context before attention when using joint attention
        """
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if self.context_dim is not None:
            self.to_k_c = nn.Linear(self.context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(self.context_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.to_q_c = nn.Linear(self.context_dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only == False: # Must be passed as False
            self.to_out_c = nn.Linear(self.inner_dim, dim)

    def attend(
        self,
        x: Tensor,
        mask: Optional[Tensor]=None,
        rope: Optional[Tuple[Tensor, Optional[Union[float, Tensor]]]]=None,  # rotary position embedding for x
    ) -> Tensor:
        """
        Single attention layer

        :param x: input tensor
        :param mask: mask tensor
        :param rope: rotary position embedding
        """
        assert x.ndim == 3, f"Input must be 3D tensor, got {x.ndim}D tensor."
        assert mask is None or mask.ndim == 2, f"Mask must be 2D tensor, got {mask.ndim}D tensor."

        batch_size = x.shape[0]

        # `sample` projections.
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = mask
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, self.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        x = x.to(query.dtype)

        # linear proj
        x = self.to_out[0](x)
        # dropout
        x = self.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x

    def forward(
        self,
        x: Tensor, # noised input x
        mask: Optional[Tensor]=None, # mask
        rope: Optional[Tuple[Tensor, Optional[Union[float, Tensor]]]]=None,  # rotary position embedding for x
    ) -> Tensor:
        """
        :param x: input tensor
        :param mask: mask tensor
        :param rope: rotary position embedding for x
        :return: output tensor
        """
        return self.attend(x, mask=mask, rope=rope)

class JointAttention(Attention):
    """
    Joint attention module, adding context to the attention mechanism
    """
    def __init__(
        self,
        dim: int,
        heads: int=8,
        dim_head: int=64,
        dropout: float=0.0,
        context_dim: Optional[int]=None,
        context_pre_only: Optional[bool]=None
    ) -> None:
        """
        :param dim: input dimension
        :param heads: number of heads
        :param dim_head: dimension of each head
        :param dropout: dropout rate
        :param context_dim: context dimension.
        :param context_pre_only: whether to apply context before attention when using joint attention
        """
        super().__init__(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        assert context_dim is not None, "Context dimension must be provided for joint attention."
        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        if self.context_pre_only is not None:
            self.to_q_c = nn.Linear(context_dim, self.inner_dim)

        if self.context_pre_only == False: # Must be passed as False
            self.to_out_c = nn.Linear(self.inner_dim, dim)

    def joint_attend(
        self,
        x: Tensor,
        c: Tensor,
        mask: Optional[Tensor]=None,
        rope: Optional[Tuple[Tensor, Optional[Union[float, Tensor]]]]=None,  # rotary position embedding for x
        c_rope: Optional[Tuple[Tensor, Optional[Union[float, Tensor]]]]=None,  # rotary position embedding for c
    ) -> Tuple[Tensor, Tensor]:
        """
        :param x: input tensor
        :param c: context tensor
        :param mask: mask tensor
        :param rope: rotary position embedding for x
        :param c_rope: rotary position embedding for c
        :return: tuple of (output tensor, context tensor)
        """
        assert x.ndim == 3, f"Input must be 3D tensor, got {x.ndim}D tensor."
        assert c.ndim == 3, f"Context must be 3D tensor, got {c.ndim}D tensor."
        assert mask is None or mask.ndim == 2, f"Mask must be 2D tensor, got {mask.ndim}D tensor."

        residual = x

        batch_size = c.shape[0]

        # `sample` projections.
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # `context` projections.
        c_query = self.to_q_c(c)
        c_key = self.to_k_c(c)
        c_value = self.to_v_c(c)

        # apply rope for context and noised input independently
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            c_query = apply_rotary_pos_emb(c_query, freqs, q_xpos_scale)
            c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

        # attention
        query = torch.cat([query, c_query], dim=1)
        key = torch.cat([key, c_key], dim=1)
        value = torch.cat([value, c_value], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = F.pad(mask, (0, c.shape[1]), value=True)  # no mask for c (text)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, self.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        x = x.to(query.dtype)

        # Split the attention outputs.
        x, c = (
            x[:, : residual.shape[1]],
            x[:, residual.shape[1] :],
        )

        # linear proj
        x = self.to_out[0](x)
        # dropout
        x = self.to_out[1](x)

        if not self.context_pre_only:
            c = self.to_out_c(c)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
            # c = c.masked_fill(~mask, 0.)  # no mask for c (text)

        return x, c

    def forward( # type: ignore[override]
        self,
        x: Tensor, # noised input x
        c: Tensor, # context c
        mask: Optional[Tensor]=None, # mask
        rope: Optional[Tuple[Tensor, Optional[Union[float, Tensor]]]]=None,  # rotary position embedding for x
        c_rope: Optional[Tuple[Tensor, Optional[Union[float, Tensor]]]]=None,  # rotary position embedding for c
    ) -> Tuple[Tensor, Tensor]:
        """
        :param x: input tensor
        :param c: context tensor
        :param mask: mask tensor
        :param rope: rotary position embedding for x
        :param c_rope: rotary position embedding for c
        :return: tuple of (output tensor, context tensor)
        """
        return self.joint_attend(x, c, mask=mask, rope=rope, c_rope=c_rope)
