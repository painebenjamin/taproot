from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["pack_tensor", "unpack_tensor", "get_packed_image_ids"]

def get_packed_image_ids(height: int, width: int) -> torch.Tensor:
    """
    Returns a tensor of shape (height * width, 3) where each row is a unique
    identifier for a pixel in a (height, width) image.
    """
    import torch

    image_ids = torch.zeros(height, width, 3)
    image_ids[..., 1] = image_ids[..., 1] + torch.arange(height)[:, None]
    image_ids[..., 2] = image_ids[..., 2] + torch.arange(width)[None, :]

    image_id_height, image_id_width, image_id_channels = image_ids.shape

    image_ids = image_ids.reshape(
        image_id_height * image_id_width, image_id_channels
    )

    return image_ids

def pack_tensor(
    tensor: torch.Tensor,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Packs (b, c, h, w) tensor into (b, (w / 2) * (h / 2), c * 4) tensor.

    >>> import torch
    >>> tensor = torch.randn(1, 3, 64, 64) # one RGB 64x64 channel
    >>> packed = pack_tensor(tensor, 1, 3, 64, 64)
    >>> packed.shape
    torch.Size([1, 1024, 12])
    """
    tensor = tensor.view(batch_size, channels, height // 2, 2, width // 2, 2)
    tensor = tensor.permute(0, 2, 4, 1, 3, 5)
    tensor = tensor.reshape(batch_size, (height // 2) * (width // 2), channels * 4)
    return tensor

def unpack_tensor(
    tensor: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Unpacks (b, (w / 2) * (h / 2), c * 4) tensor into (b, c, h, w) tensor.

    >>> import torch
    >>> tensor = torch.randn(1, 1024, 12) # packed tensor
    >>> unpacked = unpack_tensor(tensor, 64, 64)
    >>> unpacked.shape
    torch.Size([1, 3, 64, 64])
    """
    batch_size, num_patches, channels = tensor.shape
    tensor = tensor.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    tensor = tensor.permute(0, 3, 1, 4, 2, 5)
    tensor = tensor.reshape(batch_size, channels // 4, height, width)
    return tensor
