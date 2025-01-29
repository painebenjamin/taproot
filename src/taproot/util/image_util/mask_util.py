from __future__ import annotations

from typing import Union, List, TYPE_CHECKING, overload

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = [
    "dilate_erode",
    "create_mask",
]

@overload
def dilate_erode(image: Image, value: int) -> Image: ...

@overload
def dilate_erode(image: List[Image], value: int) -> List[Image]: ...

def dilate_erode(
    image: Union[Image, List[Image]],
    value: int
) -> Union[Image, List[Image]]:
    """
    Given an image, dilate or erode it.
    Values of >0 dilate, <0 erode. 0 Does nothing.
    :see: http://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
    """
    if value == 0:
        return image
    if isinstance(image, list):
        return [
            dilate_erode(img, value)
            for img in image
        ]

    from PIL import Image
    import cv2 # type: ignore[import-not-found,unused-ignore]
    import numpy as np

    arr = np.array(image.convert("L"))
    transform = cv2.dilate if value > 0 else cv2.erode
    value = abs(value)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    arr = transform(arr, kernel, iterations=1)
    return Image.fromarray(arr)

def create_mask(
    width: int,
    height: int,
    left: int,
    top: int,
    right: int,
    bottom: int
) -> Image:
    """
    Creates a mask from 6 dimensions
    """
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    draw.rectangle(((left, top), (right, bottom)), fill="#ffffff")
    return image
