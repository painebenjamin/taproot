from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = [
    "images_are_equal",
]

def images_are_equal(image_1: Image, image_2: Image) -> bool:
    """
    Determines if two images are equal.
    """
    from PIL import ImageChops
    if image_1.height != image_2.height or image_1.width != image_2.width:
        return False
    if image_1.mode == image_2.mode == "RGBA":
        image_1_alpha = [p[3] for p in image_1.getdata()] # type: ignore[attr-defined, unused-ignore]
        image_2_alpha = [p[3] for p in image_2.getdata()] # type: ignore[attr-defined, unused-ignore]
        if image_1_alpha != image_2_alpha:
            return False
    return not ImageChops.difference(
        image_1.convert("RGB"), image_2.convert("RGB")
    ).getbbox()
