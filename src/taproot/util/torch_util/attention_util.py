# Adapted from https://github.com/Wan-Video/Wan2.1/tree/main
# Licensed under Apache 2.0. Copyright 2025 The Alibaba Wan Team Authors.
from __future__ import annotations

from typing import Optional, Tuple, Union, TYPE_CHECKING
from warnings import warn

from .dtype_util import get_torch_dtype

if TYPE_CHECKING:
    import torch

__all__ = [
    "flash_attn_3_available",
    "flash_attn_2_available",
    "flash_attn_available",
    "flash_attention",
    "attention",
]

FLASH_ATTN_3_AVAILABLE: Optional[bool] = None
FLASH_ATTN_2_AVAILABLE: Optional[bool] = None

def flash_attn_3_available() -> bool:
    """
    :return: whether flash attention 3 is available
    """
    global FLASH_ATTN_3_AVAILABLE
    if FLASH_ATTN_3_AVAILABLE is None:
        try:
            import flash_attn_interface # type: ignore[import-untyped,import-not-found,unused-ignore]
            FLASH_ATTN_3_AVAILABLE = flash_attn_interface is not None
        except ModuleNotFoundError:
            FLASH_ATTN_3_AVAILABLE = False
    return FLASH_ATTN_3_AVAILABLE

def flash_attn_2_available() -> bool:
    """
    :return: whether flash attention 2 is available
    """
    global FLASH_ATTN_2_AVAILABLE
    if FLASH_ATTN_2_AVAILABLE is None:
        try:
            import flash_attn # type: ignore[import-untyped,import-not-found,unused-ignore]
            FLASH_ATTN_2_AVAILABLE = flash_attn is not None
        except ModuleNotFoundError:
            FLASH_ATTN_2_AVAILABLE = False
    return FLASH_ATTN_2_AVAILABLE

def flash_attn_available() -> bool:
    """
    :return: whether flash attention is available
    """
    return flash_attn_3_available() or flash_attn_2_available()

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    dropout_p: float = 0.,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool=False,
    window_size: Tuple[int, int]=(-1, -1),
    deterministic: bool=False,
    dtype: Union[str, torch.dtype]="bfloat16",
    version: Optional[int]=None,
) -> torch.Tensor:
    """
    :param q: query tensor, shape [batch, seq_len_q, dim]
    :param k: key tensor, shape [batch, seq_len_k, dim]
    :param v: value tensor, shape [batch, seq_len_k, dim]
    :param q_lens: query sequence lengths, shape [batch]
    :param k_lens: key sequence lengths, shape [batch]
    :param dropout_p: dropout probability
    :param softmax_scale: scale factor for softmax
    :param q_scale: scale factor for query
    :param causal: whether to use causal attention
    :param window_size: window size for local attention
    :param deterministic: whether to use deterministic attention
    :param dtype: data type for computation
    :param version: version of flash attention, 2 or 3
    :return: output tensor, shape [batch, seq_len_q, dim]
    """
    import torch
    dtype = get_torch_dtype(dtype)
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x: torch.Tensor) -> torch.Tensor:
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not flash_attn_3_available():
        warn("Flash attention 3 is not available, use flash attention 2 instead.")

    # apply attention
    if (version is None or version == 3) and flash_attn_3_available():
        import flash_attn_interface
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([
                q_lens.new_zeros([1]),
                q_lens
            ]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([
                k_lens.new_zeros([1]),
                k_lens
            ]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic
        )[0].unflatten(0, (b, lq))
    elif flash_attn_2_available():
        import flash_attn
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([
                q_lens.new_zeros([1]),
                q_lens
            ]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([
                k_lens.new_zeros([1]),
                k_lens
            ]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic
        ).unflatten(0, (b, lq))
    else:
        raise RuntimeError("Flash attention is not available.")

    # output
    return x.type(out_dtype) # type: ignore[no-any-return]

def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor]=None,
    k_lens: Optional[torch.Tensor]=None,
    dropout_p: float=0.,
    softmax_scale: Optional[float]=None,
    q_scale: Optional[float]=None,
    causal: bool=False,
    window_size: Tuple[int, int]=(-1, -1),
    deterministic: bool=False,
    dtype: Union[str, torch.dtype]="bfloat16",
    fa_version: Optional[int]=None,
) -> torch.Tensor:
    """
    :param q: query tensor, shape [batch, seq_len_q, dim]
    :param k: key tensor, shape [batch, seq_len_k, dim]
    :param v: value tensor, shape [batch, seq_len_k, dim]
    :param q_lens: query sequence lengths, shape [batch]
    :param k_lens: key sequence lengths, shape [batch]
    :param dropout_p: dropout probability
    :param softmax_scale: scale factor for softmax
    :param q_scale: scale factor for query
    :param causal: whether to use causal attention
    :param window_size: window size for local attention
    :param deterministic: whether to use deterministic attention
    :param dtype: data type for computation
    :param fa_version: version of flash attention, 2 or 3
    :return: output tensor, shape [batch, seq_len_q, dim]
    """
    if flash_attn_available():
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        import torch
        dtype = get_torch_dtype(dtype)
        if q_lens is not None or k_lens is not None:
            warn(
                "Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance."
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=causal,
            dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous()
        return out
