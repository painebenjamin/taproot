# Adapted from https://github.com/Wan-Video/Wan2.1/
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import torch
import torch.amp as amp
import torch.nn as nn

from typing import Tuple, List, Optional, Type
from typing_extensions import Literal

from taproot.util import flash_attention

WAN_MODEL_TYPE_LITERAL = Literal["t2v", "i2v"]

__all__ = ["WanModel"]

class WanRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, dim: int, eps: float=1e-5) -> None:
        """
        :param dim: The input dimension.
        :param eps: The epsilon value for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: The input tensor with shape [B, L, C].
        :return: The normalized tensor with shape [B, L, C].
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: The input tensor with shape [B, L, C].
        :return: The normalized tensor with shape [B, L, C].
        """
        return self._norm(x.float()).type_as(x) * self.weight

class WanLayerNorm(nn.LayerNorm):
    """
    Forced-FP32 Layer Normalization with a learnable scale parameter.
    """
    def __init__(
        self,
        dim: int,
        eps: float=1e-6,
        elementwise_affine: bool=False
    ) -> None:
        """
        :param dim: The input dimension.
        :param eps: The epsilon value for numerical stability.
        :param elementwise_affine: Whether to use learnable scale parameter.
        """
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: The input tensor with shape [B, L, C].
        :return: The normalized tensor with shape [B, L, C].
        """
        return super().forward(x.float()).type_as(x)

class WanSelfAttention(nn.Module):
    """
    Self-Attention module with rotary position encoding.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int]=(-1, -1),
        qk_norm: bool=True,
        eps: float=1e-6
    ) -> None:
        """
        :param dim: The input dimension.
        :param num_heads: The number of attention heads.
        :param window_size: The window size for local attention.
        :param qk_norm: Whether to apply query/key normalization.
        :param eps: The epsilon value for numerical stability.
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    @amp.autocast("cuda", enabled=False) # type: ignore[attr-defined,misc]
    def apply_rope(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: The input tensor with shape [B, L, C].
        :param grid_sizes: The grid sizes for each sample with shape [B, 3].
        :param freqs: The rope frequencies with shape [1024, C / num_heads / 2].
        :return: The tensor with shape [B, L, C].
        """
        n, c = x.size(2), x.size(3) // 2

        # split freqs
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1) # type: ignore[no-untyped-call]

        # loop over samples
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w

            # precompute multipliers
            x_i = torch.view_as_complex(
                x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
            )
            freqs_i = torch.cat(
                [
                    freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(seq_len, 1, -1)

            # apply rotary embedding
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])

            # append to collection
            output.append(x_i)
        return torch.stack(output).float()

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: The input tensor with shape [B, L, C].
        :param seq_lens: The sequence lengths for each sample with shape [B].
        :param grid_sizes: The grid sizes for each sample with shape [B, 3].
        :param freqs: The rope frequencies with shape [1024, C / num_heads / 2].
        :return: The tensor with shape [B, L, C].
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        x = flash_attention(
            q=self.apply_rope(q, grid_sizes, freqs),
            k=self.apply_rope(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanT2VCrossAttention(WanSelfAttention):
    """
    Cross-Attention module for text-to-video mode.
    """
    def forward( # type: ignore[override]
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: The input tensor with shape [B, L1, C].
        :param context: The context tensor with shape [B, L2, C].
        :param context_lens: The context lengths for each sample with shape [B].
        :return: The tensor with shape [B, L1, C].
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanI2VCrossAttention(WanSelfAttention):
    """
    Cross-Attention module for image-to-video mode.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int]=(-1, -1),
        qk_norm: bool=True,
        eps: float=1e-6
    ) -> None:
        """
        :param dim: The input dimension.
        :param num_heads: The number of attention heads.
        :param window_size: The window size for local attention.
        :param qk_norm: Whether to apply query/key normalization.
        :param eps: The epsilon value for numerical stability.
        """
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward( # type: ignore[override]
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: The input tensor with shape [B, L1, C].
        :param context: The context tensor with shape [B, L2, C].
        :param context_lens: The context lengths for each sample with shape [B].
        :return: The tensor with shape [B, L1, C].
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x

def get_wan_crossattention_class(
    model_type: WAN_MODEL_TYPE_LITERAL
) -> Type[nn.Module]:
    """
    Get the cross-attention module based on the type.

    :param cross_attn_type: The type of cross-attention module.
    :return: The cross-attention module.
    """
    if model_type == "t2v":
        return WanT2VCrossAttention
    elif model_type == "i2v":
        return WanI2VCrossAttention
    else:
        raise ValueError(f"Invalid cross-attention type: {model_type}")

class WanAttentionBlock(nn.Module):
    """
    Attention block with self-attention and cross-attention.
    """
    def __init__(
        self,
        model_type: WAN_MODEL_TYPE_LITERAL,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int]=(-1, -1),
        qk_norm: bool=True,
        cross_attn_norm: bool=False,
        eps: float=1e-6,
    ) -> None:
        """
        :param cross_attn_type: The type of cross-attention module.
        :param dim: The input dimension.
        :param ffn_dim: The intermediate dimension in feed-forward network.
        :param num_heads: The number of attention heads.
        :param window_size: The window size for local attention.
        :param qk_norm: Whether to apply query/key normalization.
        :param cross_attn_norm: Whether to apply cross-attention normalization.
        :param eps: The epsilon value for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(
            dim=dim,
            eps=eps
        )
        self.self_attn = WanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps
        )
        self.norm3 = (
            WanLayerNorm(
                dim=dim,
                eps=eps,
                elementwise_affine=True
            )
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = get_wan_crossattention_class(model_type)(
            dim=dim,
            num_heads=num_heads,
            window_size=(-1, -1),
            qk_norm=qk_norm,
            eps=eps
        )
        self.norm2 = WanLayerNorm(
            dim=dim,
            eps=eps
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: The input tensor with shape [B, L, C].
        :param e: The modulation tensor with shape [B, C].
        :param seq_lens: The sequence lengths for each sample with shape [B].
        :param grid_sizes: The grid sizes for each sample with shape [B, 3].
        :param freqs: The rope frequencies with shape [1024, C / num_heads / 2].
        :param context: The context tensor with shape [B, L2, C].
        :param context_lens: The context lengths for each sample with shape [B].
        :return: The tensor with shape [B, L, C].
        """
        with amp.autocast("cuda", dtype=torch.float32): # type: ignore[attr-defined]
            e = (self.modulation + e).chunk(6, dim=1) # type: ignore[assignment]

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs
        )

        with amp.autocast("cuda", dtype=torch.float32): # type: ignore[attr-defined]
            x = x + y * e[2]

        # cross-attention & ffn function
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])

        with amp.autocast("cuda", dtype=torch.float32): # type: ignore[attr-defined]
            x = x + y * e[5]

        return x

class Head(nn.Module):
    """
    Head module for the diffusion model.
    """
    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: Tuple[int, int, int],
        eps: float=1e-6
    ) -> None:
        """
        :param dim: The input dimension.
        :param out_dim: The output dimension.
        :param patch_size: The patch size for video embedding.
        :param eps: The epsilon value for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        :param x: The input tensor with shape [B, L, C].
        :param e: The modulation tensor with shape [B, C].
        :return: The tensor with shape [B, L, C].
        """
        with amp.autocast("cuda", dtype=torch.float32): # type: ignore[attr-defined]
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1) # type: ignore[assignment]
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])

        return x

class MLPProj(nn.Module):
    """
    Multi-Layer Perceptron projection for image embeddings.
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        """
        :param in_dim: The input dimension.
        :param out_dim: The output dimension.
        """
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        :param image_embeds: The input image embeddings with shape [B, C].
        :return: The projected image embeddings with shape [B, C].
        """
        return self.proj(image_embeds) # type: ignore[no-any-return]

class WanModel(nn.Module):
    """
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """
    def __init__(
        self,
        model_type: WAN_MODEL_TYPE_LITERAL="t2v",
        patch_size: Tuple[int, int, int]=(1, 2, 2),
        text_len: int=512,
        in_dim: int=16,
        dim: int=2048,
        ffn_dim: int=8192,
        freq_dim: int=256,
        text_dim: int=4096,
        out_dim: int=16,
        num_heads: int=16,
        num_layers: int=32,
        window_size: Tuple[int, int]=(-1, -1),
        qk_norm: bool=True,
        cross_attn_norm: bool=True,
        eps: float=1e-6,
        init_weights: bool=False,
    ) -> None:
        """
        :param model_type: The model variant ('t2v' or 'i2v').
        :param patch_size: The 3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        :param text_len: The fixed length for text embeddings.
        :param in_dim: The input video channels (C_in).
        :param dim: The hidden dimension of the transformer.
        :param ffn_dim: The intermediate dimension in feed-forward network.
        :param freq_dim: The dimension for sinusoidal time embeddings.
        :param text_dim: The input dimension for text embeddings.
        :param out_dim: The output video channels (C_out).
        :param num_heads: The number of attention heads.
        :param num_layers: The number of transformer blocks.
        :param window_size: The window size for local attention (-1 indicates global attention).
        :param qk_norm: Whether to enable query/key normalization.
        :param cross_attn_norm: Whether to enable cross-attention normalization.
        :param eps: The epsilon value for normalization layers.
        """
        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    model_type=model_type,
                    dim=dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    qk_norm=qk_norm,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(
            dim=dim,
            out_dim=out_dim,
            patch_size=patch_size,
            eps=eps
        )

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                self.rope_params(1024, d - 4 * (d // 6)),
                self.rope_params(1024, 2 * (d // 6)),
                self.rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(
                in_dim=1280,
                out_dim=dim
            )

        if init_weights:
            # initialize weights
            self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize model parameters using Xavier initialization.
        """
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    @amp.autocast("cuda", enabled=False) # type: ignore[attr-defined,misc]
    def rope_params(
        self,
        max_seq_len: int,
        dim: int,
        theta: int=10000
    ) -> torch.Tensor:
        """
        :param max_seq_len: The maximum sequence length.
        :param dim: The dimension of the rope parameters.
        :param theta: The theta value for sinusoidal encoding.
        :return: The rope parameters with shape [max_seq_len, dim].
        """
        assert dim % 2 == 0
        freqs = torch.outer(
            torch.arange(max_seq_len),
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def sinusoidal_embedding_1d(
        self,
        dim: int,
        position: torch.Tensor
    ) -> torch.Tensor:
        """
        :param dim: The dimension of the sinusoidal embedding.
        :param position: The position tensor with shape [B].
        :return: The sinusoidal embedding with shape [B, dim].
        """
        # preprocess
        assert dim % 2 == 0
        half = dim // 2
        position = position.type(torch.float64)

        # calculation
        sinusoid = torch.outer(
            position,
            torch.pow(10000, -torch.arange(half).to(position).div(half))
        )
        x = torch.cat([
            torch.cos(sinusoid),
            torch.sin(sinusoid)
        ], dim=1)
        return x

    def unpatchify(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        :param x: The input tensor with shape [B, L, C].
        :param grid_sizes: The grid sizes for each sample with shape [B, 3].
        :return: The tensor with shape [B, C_out, F, H, W].
        """
        c = self.out_dim
        out: List[torch.Tensor] = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)

        return out

    def forward(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
        clip_fea: Optional[torch.Tensor]=None,
        y: Optional[List[torch.Tensor]]=None
    ) -> List[torch.Tensor]:
        """
        Forward pass through the diffusion model

        :param x: The input video tensors with shape [C_in, F, H, W].
        :param t: The diffusion timesteps tensor with shape [B].
        :param context: The text embeddings with shape [L, C].
        :param seq_len: The maximum sequence length for positional encoding.
        :param clip_fea: The CLIP image features for image-to-video mode.
        :param y: The conditional video inputs for image-to-video mode.
        :return: The denoised video tensors with original input shapes [C_out, F, H / 8, W / 8].
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [
            self.patch_embedding(u.unsqueeze(0))
            for u in x
        ]
        grid_sizes = torch.stack([
            torch.tensor(u.shape[2:], dtype=torch.long)
            for u in x
        ])
        x = [
            u.flatten(2).transpose(1, 2)
            for u in x
        ]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([ # type: ignore[assignment]
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32): # type: ignore[attr-defined]
            e = self.time_embedding(
                self.sinusoidal_embedding_1d(self.freq_dim, t).float()
            )
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1) # type: ignore[assignment,list-item]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes) # type: ignore[arg-type]
        return [u.float() for u in x]
