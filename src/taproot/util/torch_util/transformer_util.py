from __future__ import annotations

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "rotate_half",
    "apply_rotary_pos_emb",
    "get_pos_embed_indices",
    "precompute_freqs_cis",
]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half of a tensor.

    :param x: the input tensor
    :return: the tensor with half of it rotated
    """
    import torch
    from einops import rearrange
    x = rearrange(x, "... (d r) -> ... d r", r = 2)
    x_1, x_2 = x.unbind(dim = -1)
    x = torch.stack((-x_2, x_1), dim = -1)
    return rearrange(x, "... d r -> ... (d r)")

def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    scale: Union[torch.Tensor, float]=1.0
) -> torch.Tensor:
    """
    Apply rotary position embeddings to a tensor.

    :param t: the input tensor
    :param freqs: the frequencies tensor
    :param scale: the scale factor
    :return: the tensor with the rotary position embeddings applied
    :see: https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
    """
    import torch
    rot_dim = freqs.shape[-1]
    seq_len = t.shape[-2]
    orig_dtype = t.dtype

    freqs = freqs[-seq_len:, :]
    scale = scale.mean().item() if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t, t_unrotated), dim = -1)

    return out.type(orig_dtype)

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float=10000.0, 
    theta_rescale_factor: float=1.0
) -> torch.Tensor:
    """
    Compute scaled rotary embeddings to longer sequence length without fine-tuning

    :param dim: the dimensionality of the tensor
    :param end: the end of the tensor
    :param theta: the theta value
    :param theta_rescale_factor: the theta rescale factor
    :return: the tensor with the scaled rotary embeddings
    :see https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have:
    :see https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py:
    """
    import torch
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(
    start: torch.Tensor,
    length: int,
    max_pos: float,
    scale: Union[float, torch.Tensor]=1.0
) -> torch.Tensor:
    """
    Get position embedding indices.
    
    :param start: the start tensor
    :param length: the length of the tensor
    :param max_pos: the maximum position
    :param scale: the scale factor
    :return: the tensor with the position embedding indices
    """
    import torch
    s = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (
            torch.arange(
                length,
                device=start.device,
                dtype=torch.float32
            ).unsqueeze(0) * s.unsqueeze(1)
        ).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos
