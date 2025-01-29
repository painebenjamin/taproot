from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = [
    "image_pixelize",
]

def image_pixelize(
    image: Image,
    factor: int = 2,
    exact: bool = True
) -> Image:
    """
    Makes an image pixelized by downsizing and upsizing by a factor.
    """
    from PIL import Image
    from PIL.Image import Resampling
    width, height = image.size
    downsample_width = width // 2 ** factor
    downsample_height = height // 2 ** factor
    upsample_width = downsample_width * 2 ** factor if exact else width
    upsample_height = downsample_height * 2 ** factor if exact else height
    image = image.resize((downsample_width, downsample_height), resample=Resampling.NEAREST)
    image = image.resize((upsample_width, upsample_height), resample=Resampling.NEAREST)
    return image
