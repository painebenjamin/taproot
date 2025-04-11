# Adapted from https://github.com/Wan-Video/Wan2.1
# Also adapted from https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/wanvideo/wan_video_vae.py
# Licensed under Apache 2.0
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Optional, Union, List, Tuple, Sequence
from typing_extensions import Literal

from einops import rearrange, repeat
from taproot.util import maybe_use_tqdm

__all__ = ["WanVAE", "WanVideoVAE"]

CACHE_T = 2

class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolution.
    """
    _padding: Tuple[int, int, int, int, int, int]
    padding: Tuple[int, int, int]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(
        self,
        x: torch.Tensor,
        cache_x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        :param x: input tensor [B, C, T, H, W]
        :param cache_x: cached input tensor [B, C, T', H, W]
        :return: output tensor [B, C_out, T, H, W]
        """
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)

class RMSNorm(nn.Module):
    """
    Root mean square layer normalization.
    """
    def __init__(
        self,
        dim: int,
        channel_first: bool=True,
        images: bool=True,
        bias: bool=False
    ) -> None:
        """
        :param dim: input dimension
        :param channel_first: whether the input is channel first
        :param images: whether the input is images
        :param bias: whether to use bias
        """
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        :return: normalized tensor
        """
        return ( # type: ignore[no-any-return]
            F.normalize(x, dim=(1 if self.channel_first else -1))
            * self.scale
            * self.gamma
            + self.bias
        )

class Upsample(nn.Upsample):
    """
    Upsample layer with nearest neighbor interpolation.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)

RESAMPLE_MODE_LITERAL = Literal["none", "upsample2d", "upsample3d", "downsample2d", "downsample3d"]

class Resample(nn.Module):
    """
    Resample layer for 2d and 3d.
    """
    resample: nn.Module

    def __init__(
        self,
        dim: int,
        mode: Optional[RESAMPLE_MODE_LITERAL]=None
    ) -> None:
        """
        :param dim: input dimension
        :param mode: resample mode
        """
        assert mode in (
            None,
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        # resample layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        else:
            self.resample = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[Sequence[Union[str, torch.Tensor]]]=None,
        feat_idx: List[int]=[0]
    ) -> torch.Tensor:
        """
        :param x: input tensor [B, C, T, H, W]
        :param feat_cache: feature cache
        :param feat_idx: feature index
        :return: output tensor [B, C, T, H, W]
        """
        b, c, t, h, w = x.size()

        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep" # type: ignore[index]
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()

                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] != "Rep"
                    ):
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :] # type: ignore[union-attr,index]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )

                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] == "Rep"
                    ):
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )

                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])

                    feat_cache[idx] = cache_x # type: ignore[index]
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone() # type: ignore[index]
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2) # type: ignore[list-item,index]
                    )
                    feat_cache[idx] = cache_x # type: ignore[index]
                    feat_idx[0] += 1
        return x

class ResidualBlock(nn.Module):
    """
    Residual block with causal convolutions.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float=0.0
    ) -> None:
        """
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param dropout: dropout rate
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[Sequence[Union[str, torch.Tensor]]]=None,
        feat_idx: List[int]=[0]
    ) -> torch.Tensor:
        """
        :param x: input tensor [B, C, T, H, W]
        :param feat_cache: feature cache
        :param feat_idx: feature index
        :return: output tensor [B, C, T, H, W]
        """
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and isinstance(feat_cache[idx], torch.Tensor):
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :] # type: ignore[union-attr,index]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x # type: ignore[index]
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x + h # type: ignore[no-any-return]

class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """
    def __init__(self, dim: int) -> None:
        """
        :param dim: input dimension
        """
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor [B, C, T, H, W]
        :return: output tensor [B, C, T, H, W]
        """
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        # compute query, key, value
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity

class Encoder3d(nn.Module):
    """
    3d encoder with causal convolutions.
    """
    def __init__(
        self,
        dim: int=128,
        z_dim: int=4,
        dim_mult: List[int]=[1, 2, 4, 4],
        num_res_blocks: int=2,
        attn_scales: List[float]=[],
        temporal_downsample: List[bool]=[True, True, False],
        dropout: float=0.0
    ) -> None:
        """
        :param dim: input dimension
        :param z_dim: output dimension
        :param dim_mult: dimension multiplier
        :param num_res_blocks: number of residual blocks
        :param attn_scales: attention scales
        :param temporal_downsample: temporal downsample toggle
        :param dropout: dropout rate
        """
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples: List[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))

                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temporal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode)) # type: ignore[arg-type]
                scale /= 2.0

        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # output blocks
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[Sequence[Union[str, torch.Tensor]]]=None,
        feat_idx: List[int]=[0]
    ) -> torch.Tensor:
        """
        :param x: input tensor [B, C, T, H, W]
        :param feat_cache: feature cache
        :param feat_idx: feature index
        :return: output tensor [B, C, T, H, W]
        """
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and isinstance(feat_cache[idx], torch.Tensor):
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), # type: ignore[union-attr,index]
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x # type: ignore[index]
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and isinstance(feat_cache[idx], torch.Tensor):
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :] # type: ignore[union-attr,index]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x # type: ignore[index]
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x

class Decoder3d(nn.Module):
    """
    3d decoder with causal convolutions.
    """
    def __init__(
        self,
        dim: int=128,
        z_dim: int=4,
        dim_mult: List[int]=[1, 2, 4, 4],
        num_res_blocks: int=2,
        attn_scales: List[float]=[],
        temporal_upsample: List[bool]=[False, True, True],
        dropout: float=0.0,
    ) -> None:
        """
        :param dim: input dimension
        :param z_dim: output dimension
        :param dim_mult: dimension multiplier
        :param num_res_blocks: number of residual blocks
        :param attn_scales: attention scales
        :param temporal_upsample: temporal upsample toggle
        :param dropout: dropout rate
        """
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_upsample = temporal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # upsample blocks
        upsamples: List[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temporal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode)) # type: ignore[arg-type]
                scale *= 2.0

        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[Sequence[Union[str, torch.Tensor]]]=None,
        feat_idx: List[int]=[0]
    ) -> torch.Tensor:
        """
        :param x: input tensor [B, C, T, H, W]
        :param feat_cache: feature cache
        :param feat_idx: feature index
        :return: output tensor [B, C, T, H, W]
        """
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and isinstance(feat_cache[idx], torch.Tensor):
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), # type: ignore[union-attr,index]
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x # type: ignore[index]
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and isinstance(feat_cache[idx], torch.Tensor):
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :] # type: ignore[union-attr,index]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x # type: ignore[index]
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class WanVAE(nn.Module):
    """
    Video VAE model.
    """
    def __init__(
        self,
        dim: int=128,
        z_dim: int=4,
        dim_mult: List[int]=[1, 2, 4, 4],
        num_res_blocks: int=2,
        attn_scales: List[float]=[],
        temporal_downsample: List[bool]=[True, True, False],
        dropout: float=0.0,
    ) -> None:
        """
        :param dim: input dimension
        :param z_dim: output dimension
        :param dim_mult: dimension multiplier
        :param num_res_blocks: number of residual blocks
        :param attn_scales: attention scales
        :param temporal_downsample: temporal downsample toggle
        :param dropout: dropout rate
        """
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample
        self.temporal_upsample = temporal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temporal_downsample,
            dropout,
        )
        self.decoder = Decoder3d(
            dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temporal_upsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.encoder_num_conv = self.count_conv3d_in_module(self.encoder)
        self.decoder_num_conv = self.count_conv3d_in_module(self.decoder)

    def count_conv3d_in_module(self, model: nn.Module) -> int:
        """
        Count the number of 3d convolutional layers in a module.

        :param model: input module
        :return: number of 3d convolutional layers
        """
        count = 0
        for m in model.modules():
            if isinstance(m, CausalConv3d):
                count += 1
        return count

    def encode(
        self,
        x: torch.Tensor,
        scale: Union[Tuple[torch.Tensor, torch.Tensor], Tuple[float, float]],
    ) -> torch.Tensor:
        """
        :param x: input tensor [B, C, T, H, W]
        :param scale: scale tensor
        :return: output tensor [B, C, T, H, W]
        """
        self.clear_cache()
        t = x.shape[2]
        num_iter = 1 + (t - 1) // 4

        for i in range(num_iter):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)

        mu, log_var = self.conv1(out).chunk(2, dim=1)

        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(mu.device, dtype=mu.dtype) for s in scale] # type: ignore[assignment]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            mu = (mu - scale[0]) * scale[1]

        self.clear_cache()
        return mu # type: ignore[no-any-return]

    def decode(
        self,
        z: torch.Tensor,
        scale: Union[Tuple[torch.Tensor, torch.Tensor], Tuple[float, float]],
        loop: bool=False
    ) -> torch.Tensor:
        """
        :param z: input tensor [B, C, T, H, W]
        :param scale: scale tensor
        :return: output tensor [B, C, T, H, W]
        """
        self.clear_cache()

        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(z.device, dtype=z.dtype) for s in scale] # type: ignore[assignment]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]

        num_iter = z.shape[2]
        x = self.conv2(z)
        for i in range(num_iter):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)

        self.clear_cache()
        return out # type: ignore[no-any-return]

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        :param mu: mean tensor
        :param log_var: log variance tensor
        :return: output tensor
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def clear_cache(self) -> None:
        """
        Clear cache.
        """
        self._conv_idx = [0]
        self._feat_map = [None] * self.decoder_num_conv
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self.encoder_num_conv

class WanVideoVAE(nn.Module):
    """
    Video VAE model.
    """
    scale: Tuple[torch.Tensor, torch.Tensor]

    def __init__(
        self,
        dim: int=96,
        z_dim: int=16,
        dim_mult: List[int]=[1, 2, 4, 4],
        num_res_blocks: int=2,
        attn_scales: List[float]=[],
        temporal_downsample: List[bool]=[False,True,True],
        dropout: float=0.0,
        stride: Tuple[int, int, int]=(4, 8, 8),
        mean: List[float]=[
            -0.7571, -0.7089, -0.9113, 0.1075,
            -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632,
            -0.1922, -0.9497, 0.2503, -0.2921, 
        ],
        std: List[float]=[
            2.8184, 1.4541, 2.3275, 2.6558,
            1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579,
            1.6382, 1.1253, 2.8251, 1.9160,
        ],
        upsampling_factor: int=8
    ) -> None:
        """
        :param dim: input dimension
        :param z_dim: output dimension
        :param dim_mult: dimension multiplier
        :param num_res_blocks: number of residual blocks
        :param attn_scales: attention scales
        :param temporal_downsample: temporal downsample toggle
        :param dropout: dropout rate
        :param mean: means
        :param std: stds
        """
        super().__init__()
        self.model = WanVAE(
            dim=dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temporal_downsample=temporal_downsample,
            dropout=dropout,
        )
        self.stride = stride
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = (self.mean, 1.0 / self.std)
        self.upsampling_factor = upsampling_factor

    def get_target_shape(
        self,
        num_frames: int,
        height: int,
        width: int
    ) -> Tuple[int, int, int, int]:
        """
        Gets the target shape based on the input shape.

        :param num_frames: number of frames
        :param height: height in pixels
        :param width: width in pixels
        :return: target shape [C, T, H, W]
        """
        return (
            self.model.z_dim,
            (num_frames - 1) // self.stride[0] + 1,
            height // self.stride[1],
            width // self.stride[2],
        )

    def build_1d_mask(
        self,
        length: int,
        left_bound: bool,
        right_bound: bool,
        border_width: int
    ) -> torch.Tensor:
        """
        Builds a 1D mask.

        :param length: length
        :param left_bound: left bound
        :param right_bound: right bound
        :param border_width: border width
        :return: mask
        """
        x = torch.ones((length,))

        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))

        return x

    def build_mask(
        self,
        data: torch.Tensor,
        is_bound: Tuple[bool, bool, bool, bool],
        border_width: Tuple[int, int],
    ) -> torch.Tensor:
        """
        :param data: data tensor [B, C, T, H, W]
        :param is_bound: bound toggle for each side
        :param border_width: border width
        :return: mask tensor [1, 1, 1, H, W]
        """
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    def tiled_decode(
        self,
        hidden_states: torch.Tensor,
        device: torch.device,
        tile_size: Tuple[int, int],
        tile_stride: Tuple[int, int],
        loop: bool,
        use_tqdm: bool=True
    ) -> torch.Tensor:
        """
        :param hidden_states: hidden states tensor [B, C, T, H, W]
        :param device: device
        :param tile_size: tile size
        :param tile_stride: tile stride
        :return: output tensor [B, C, T, H, W]
        """
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if h-stride_h >= 0 and h-stride_h+size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w-stride_w >= 0 and w-stride_w+size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = T * 4 - 3
        weight = torch.zeros((1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)
        values = torch.zeros((1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)

        for h, h_, w, w_ in maybe_use_tqdm(tasks, desc="VAE decoding", use_tqdm=use_tqdm):
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.decode(hidden_states_batch, self.scale, loop).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) * self.upsampling_factor, (size_w - stride_w) * self.upsampling_factor)
            ).to(dtype=hidden_states.dtype, device=data_device)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        values = values.float().clamp_(-1, 1)
        return values

    def tiled_encode(
        self,
        video: torch.Tensor,
        device: torch.device,
        tile_size: Tuple[int, int],
        tile_stride: Tuple[int, int],
        use_tqdm: bool=True
    ) -> torch.Tensor:
        """
        :param video: video tensor [B, C, T, H, W]
        :param device: device
        :param tile_size: tile size
        :param tile_stride: tile stride
        :return: output tensor [B, C, T, H, W]
        """
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if (h-stride_h >= 0 and h-stride_h+size_h >= H):
                continue
            for w in range(0, W, stride_w):
                if (w-stride_w >= 0 and w-stride_w+size_w >= W):
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = (T + 3) // 4
        weight = torch.zeros((1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)
        values = torch.zeros((1, 16, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)

        for h, h_, w, w_ in maybe_use_tqdm(tasks, desc="VAE encoding", use_tqdm=use_tqdm):
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) // self.upsampling_factor, (size_w - stride_w) // self.upsampling_factor)
            ).to(dtype=video.dtype, device=data_device)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        return values.to(device)

    def single_encode(
        self,
        video: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        :param video: video tensor [B, C, T, H, W]
        :param device: device
        :return: output tensor [B, C, T, H, W]
        """
        video = video.to(device)
        x = self.model.encode(video, self.scale)
        return x

    def single_decode(
        self,
        hidden_states: torch.Tensor,
        device: torch.device,
        loop: bool
    ) -> torch.Tensor:
        """
        :param hidden_states: hidden states tensor [B, C, T, H, W]
        :param device: device
        :return: output tensor [B, C, T, H, W]
        """
        hidden_states = hidden_states.to(device)
        video = self.model.decode(hidden_states, self.scale, loop)
        return video.float().clamp_(-1, 1)

    @torch.no_grad()
    def encode(
        self,
        videos: List[torch.Tensor],
        device: torch.device,
        tiled: bool=False,
        tile_size: Tuple[int, int]=(66, 66),
        tile_stride: Tuple[int, int]=(34, 32)
    ) -> torch.Tensor:
        """
        :param videos: list of video tensors [C, T, H, W]
        :param device: device
        :param tiled: tiled encoding
        :param tile_size: tile size
        :param tile_stride: tile stride
        :return: output tensor [B, C, T, H, W]
        """
        videos = [video.to("cpu") for video in videos]
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                tile_size = (tile_size[0], tile_size[1])
                tile_stride = (tile_stride[0], tile_stride[1])
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride)
            else:
                hidden_state = self.single_encode(video, device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)

        return torch.stack(hidden_states)

    @torch.no_grad()
    def decode(
        self,
        hidden_states: List[torch.Tensor],
        device: torch.device,
        tiled: bool=False,
        tile_size: Tuple[int, int]=(66, 66),
        tile_stride: Tuple[int, int]=(34, 32),
        loop: bool=False
    ) -> List[torch.Tensor]:
        """
        :param hidden_states: list of hidden states tensors [B, C, T, H, W]
        :param device: device
        :param tiled: tiled decoding
        :param tile_size: tile size
        :param tile_stride: tile stride
        :return: list of output tensors [B, C, T, H, W]
        """
        hidden_states = [hidden_state.to("cpu") for hidden_state in hidden_states]
        videos = []
        for hidden_state in hidden_states:
            hidden_state = hidden_state.unsqueeze(0)
            if tiled:
                video = self.tiled_decode(hidden_state, device, tile_size, tile_stride, loop)
            else:
                video = self.single_decode(hidden_state, device, loop)
            video = video.squeeze(0)
            videos.append(video)

        return videos
