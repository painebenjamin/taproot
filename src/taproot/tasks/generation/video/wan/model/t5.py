# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union

from taproot.util import float16_finite_clamp

__all__ = [
    "T5Model",
    "T5Encoder",
    "T5Decoder",
]

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, L, C].
        :return: [B, L, C].
        """
        return (
            0.5 * x * (
                1.0 + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (
                        x + 0.044715 * torch.pow(x, 3.0)
                    )
                )
            )
        )

class T5RelativeEmbedding(nn.Module):
    """
    Relative position embeddings.
    """
    def __init__(
        self,
        num_buckets: int,
        num_heads: int,
        bidirectional: bool,
        max_dist: int=128
    ) -> None:
        """
        :param num_buckets: Number of buckets.
        :param num_heads: Number of heads.
        :param bidirectional: Whether to use bidirectional embeddings.
        :param max_dist: Maximum distance.
        """
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute relative position bucket.

        :param rel_pos: Relative position tensor [B, Lq, Lk].
        :return: Relative position bucket tensor [B, Lq, Lk].
        """
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0 # type: ignore[assignment]
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = (
            max_exact
            + (
                torch.log(rel_pos.float() / max_exact)
                / math.log(self.max_dist / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1)
        )
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)

        return rel_buckets

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        """
        :param lq: Length of query.
        :param lk: Length of key.
        :return: Relative position embeddings [N, Lq, Lk].
        """
        device = self.embedding.weight.device
        rel_pos = (
            torch.arange(lk, device=device).unsqueeze(0) - 
            torch.arange(lq, device=device).unsqueeze(1)
        )
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(0)  # [1, N, Lq, Lk]

        return rel_pos_embeds.contiguous() # type: ignore[no-any-return]

class T5LayerNorm(nn.Module):
    """
    Layer normalization (https://arxiv.org/abs/1607.06450).
    """
    def __init__(self, dim: int, eps: float=1e-6) -> None:
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, L, C].
        :return: [B, L, C].
        """
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)

        return self.weight * x

class T5Attention(nn.Module):
    """
    Multi-head attention layer.
    """
    def __init__(
        self,
        dim: int,
        dim_attn: int,
        num_heads: int,
        dropout: float=0.1
    ) -> None:
        """
        :param dim: Dimension of input.
        :param dim_attn: Dimension of attention.
        :param num_heads: Number of heads.
        :param dropout: Dropout rate.
        """
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor]=None,
        mask: Optional[torch.Tensor]=None,
        pos_bias: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        :param x: Input tensor [B, L1, C].
        :param context: Context tensor [B, L2, C] or None.
        :param mask: Mask tensor [B, L2] or [B, L1, L2] or None.
        :param pos_bias: Positional bias tensor [B, N, L1, L2] or None.
        :return: Output tensor [B, L1, C].
        """
        # check inputs
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # attention bias
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias

        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        # compute attention (T5 does not use scaling)
        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)

        # output
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)

        return x

class T5FeedForward(nn.Module):
    """
    Feed-forward layer.
    """
    def __init__(self, dim: int, dim_ffn: int, dropout: float=0.1) -> None:
        """
        :param dim: Dimension of input.
        :param dim_ffn: Dimension of feed-forward layer.
        :param dropout: Dropout rate.
        """
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor [B, L, C].
        :return: Output tensor [B, L, C].
        """
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

class T5SelfAttention(nn.Module):
    """
    Multi-head self-attention layer.
    """
    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool=True,
        dropout: float=0.1
    ) -> None:
        """
        :param dim: Dimension of input.
        :param dim_attn: Dimension of attention.
        :param dim_ffn: Dimension of feed-forward layer.
        :param num_heads: Number of heads.
        :param num_buckets: Number of buckets for relative position.
        :param shared_pos: Whether to share positional embeddings.
        :param dropout: Dropout rate.
        """
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = (
            None
            if shared_pos
            else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor]=None,
        pos_bias: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        :param x: Input tensor [B, L, C].
        :param mask: Mask tensor [B, L] or [B, L, L] or None.
        :param pos_bias: Positional bias tensor [B, N, L, L] or None.
        :return: Output tensor [B, L, C].
        """
        e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1)) # type: ignore[misc]
        x = float16_finite_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = float16_finite_clamp(x + self.ffn(self.norm2(x)))

        return x

class T5CrossAttention(nn.Module):
    """
    Multi-head cross-attention layer.
    """
    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool=True,
        dropout: float=0.1
    ) -> None:
        """
        :param dim: Dimension of input.
        :param dim_attn: Dimension of attention.
        :param dim_ffn: Dimension of feed-forward layer.
        :param num_heads: Number of heads.
        :param num_buckets: Number of buckets for relative position.
        :param shared_pos: Whether to share positional embeddings.
        :param dropout: Dropout rate.
        """
        super(T5CrossAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm3 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = (
            None
            if shared_pos
            else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=False)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor]=None,
        encoder_states: Optional[torch.Tensor]=None,
        encoder_mask: Optional[torch.Tensor]=None,
        pos_bias: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        :param x: Input tensor [B, L, C].
        :param mask: Mask tensor [B, L] or [B, L, L] or None.
        :param encoder_states: Encoder states tensor [B, L, C].
        :param encoder_mask: Encoder mask tensor [B, L] or [B, L, L] or None.
        :param pos_bias: Positional bias tensor [B, N, L, L] or None.
        :return: Output tensor [B, L, C].
        """
        e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1)) # type: ignore[misc]
        x = float16_finite_clamp(
            x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e)
        )
        x = float16_finite_clamp(
            x
            + self.cross_attn(self.norm2(x), context=encoder_states, mask=encoder_mask)
        )
        x = float16_finite_clamp(x + self.ffn(self.norm3(x)))

        return x

class T5Module(nn.Module):
    """
    A base class for T5 modules.
    """
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """
        Initialize weights for a module.
        """
        if isinstance(m, T5LayerNorm):
            nn.init.ones_(m.weight)
        elif isinstance(m, T5Model):
            nn.init.normal_(m.token_embedding.weight, std=1.0)
        elif isinstance(m, T5FeedForward):
            nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
            nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
            nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
        elif isinstance(m, T5Attention):
            nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn) ** -0.5)
            nn.init.normal_(m.k.weight, std=m.dim**-0.5)
            nn.init.normal_(m.v.weight, std=m.dim**-0.5)
            nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn) ** -0.5)
        elif isinstance(m, T5RelativeEmbedding):
            nn.init.normal_(
                m.embedding.weight, std=(2 * m.num_buckets * m.num_heads) ** -0.5
            )

    def init_weights(self) -> None:
        """
        Initialize weights for the model.
        """
        self.apply(self._init_weights)

class T5Encoder(T5Module):
    """
    Encoder module for T5.
    """
    def __init__(
        self,
        vocab: Union[int, nn.Embedding],
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_layers: int,
        num_buckets: int,
        shared_pos: bool=True,
        init_weights: bool=False,
        dropout: float=0.1,
    ) -> None:
        """
        :param vocab: Vocabulary size or embedding layer.
        :param dim: Dimension of input.
        :param dim_attn: Dimension of attention.
        :param dim_ffn: Dimension of feed-forward layer.
        :param num_heads: Number of heads.
        :param num_layers: Number of layers.
        :param num_buckets: Number of buckets for relative position.
        :param shared_pos: Whether to share positional embeddings.
        :param init_weights: Whether to initialize weights.
        :param dropout: Dropout rate.
        """
        super(T5Encoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = (
            vocab if isinstance(vocab, nn.Embedding) else nn.Embedding(vocab, dim)
        )
        self.pos_embedding = (
            T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
            if shared_pos
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                T5SelfAttention(
                    dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = T5LayerNorm(dim)

        # initialize weights
        if init_weights:
            self.init_weights()

    def forward(
        self,
        ids: torch.Tensor,
        mask: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        :param ids: Input tensor [B, L].
        :param mask: Mask tensor [B, L] or [B, L, L] or None.
        :return: Output tensor [B, L, C].
        """
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None # type: ignore[misc]

        for block in self.blocks:
            x = block(x, mask, pos_bias=e)

        x = self.norm(x)
        x = self.dropout(x)

        return x # type: ignore[no-any-return]

class T5Decoder(T5Module):
    """
    Decoder module for T5.
    """
    def __init__(
        self,
        vocab: Union[int, nn.Embedding],
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_layers: int,
        num_buckets: int,
        shared_pos: bool=True,
        init_weights: bool=False,
        dropout: float=0.1,
    ) -> None:
        """
        :param vocab: Vocabulary size or embedding layer.
        :param dim: Dimension of input.
        :param dim_attn: Dimension of attention.
        :param dim_ffn: Dimension of feed-forward layer.
        :param num_heads: Number of heads.
        :param num_layers: Number of layers.
        :param num_buckets: Number of buckets for relative position.
        :param shared_pos: Whether to share positional embeddings.
        :param init_weights: Whether to initialize weights.
        :param dropout: Dropout rate.
        """
        super(T5Decoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = (
            vocab if isinstance(vocab, nn.Embedding) else nn.Embedding(vocab, dim)
        )
        self.pos_embedding = (
            T5RelativeEmbedding(num_buckets, num_heads, bidirectional=False)
            if shared_pos
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                T5CrossAttention(
                    dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = T5LayerNorm(dim)

        if init_weights:
            self.init_weights()

    def forward(
        self,
        ids: torch.Tensor,
        mask: Optional[torch.Tensor]=None,
        encoder_states: Optional[torch.Tensor]=None,
        encoder_mask: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        :param ids: Input tensor [B, L].
        :param mask: Mask tensor [B, L] or [B, L, L] or None.
        :param encoder_states: Encoder states tensor [B, L, C] or None.
        :param encoder_mask: Encoder mask tensor [B, L] or [B, L, L] or None.
        :return: Output tensor [B, L, C].
        """
        b, s = ids.size()

        # causal mask
        if mask is None:
            mask = torch.tril(torch.ones(1, s, s).to(ids.device))
        elif mask.ndim == 2:
            mask = torch.tril(mask.unsqueeze(1).expand(-1, s, -1))

        # layers
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e: Optional[torch.Tensor] = None

        if self.shared_pos:
            e = self.pos_embedding(x.size(1), x.size(1)) # type: ignore[misc]

        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)

        x = self.norm(x)
        x = self.dropout(x)

        return x # type: ignore[no-any-return]

class T5Model(T5Module):
    """
    Joint encoder-decoder model for T5.
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        encoder_layers: int,
        decoder_layers: int,
        num_buckets: int,
        shared_pos: bool=True,
        init_weights: bool=False,
        dropout: float=0.1
    ) -> None:
        """
        :param vocab_size: Vocabulary size.
        :param dim: Dimension of input.
        :param dim_attn: Dimension of attention.
        :param dim_ffn: Dimension of feed-forward layer.
        :param num_heads: Number of heads.
        :param encoder_layers: Number of encoder layers.
        :param decoder_layers: Number of decoder layers.
        :param num_buckets: Number of buckets for relative position.
        :param shared_pos: Whether to share positional embeddings.
        :param init_weights: Whether to initialize weights.
        :param dropout: Dropout rate.
        """
        super(T5Model, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(
            self.token_embedding,
            dim,
            dim_attn,
            dim_ffn,
            num_heads,
            encoder_layers,
            num_buckets,
            shared_pos,
            init_weights,
            dropout,
        )
        self.decoder = T5Decoder(
            self.token_embedding,
            dim,
            dim_attn,
            dim_ffn,
            num_heads,
            decoder_layers,
            num_buckets,
            shared_pos,
            init_weights,
            dropout,
        )
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # initialize weights
        if init_weights:
            self.init_weights()

    def forward(
        self,
        encoder_ids: torch.Tensor,
        decoder_ids: torch.Tensor,
        encoder_mask: Optional[torch.Tensor]=None,
        decoder_mask: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        :param encoder_ids: Encoder input tensor [B, L].
        :param decoder_ids: Decoder input tensor [B, L].
        :param encoder_mask: Encoder mask tensor [B, L] or [B, L, L] or None.
        :param decoder_mask: Decoder mask tensor [B, L] or [B, L, L] or None.
        :return: Output tensor [B, L, V].
        """
        x = self.encoder(encoder_ids, encoder_mask)
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        x = self.head(x)

        return x # type: ignore[no-any-return]
