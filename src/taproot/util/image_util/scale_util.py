from __future__ import annotations

import math

from typing import Optional, Union, List, TYPE_CHECKING, overload

from ...constants import *

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = [
    "fit_image",
    "scale_image",
]

@overload
def fit_image(
    image: Image,
    width: int,
    height: int,
    fit: Optional[IMAGE_FIT_LITERAL] = None,
    anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
    offset_left: Optional[int] = None,
    offset_top: Optional[int] = None
) -> Image: ...

@overload
def fit_image(
    image: List[Image],
    width: int,
    height: int,
    fit: Optional[IMAGE_FIT_LITERAL] = None,
    anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
    offset_left: Optional[int] = None,
    offset_top: Optional[int] = None
) -> List[Image]: ...

def fit_image(
    image: Union[Image, List[Image]],
    width: int,
    height: int,
    fit: Optional[IMAGE_FIT_LITERAL] = None,
    anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
    offset_left: Optional[int] = None,
    offset_top: Optional[int] = None
) -> Union[Image, List[Image]]:
    """
    Given an image of unknown size, make it a known size with optional fit parameters.
    """
    if not isinstance(image, list):
        if getattr(image, "n_frames", 1) > 1:
            frames = []
            for i in range(image.n_frames): # type: ignore[attr-defined,unused-ignore]
                image.seek(i)
                frames.append(image.copy().convert("RGBA"))
            image = frames
    if isinstance(image, list):
        return [
            fit_image(
                img,
                width=width,
                height=height,
                fit=fit,
                anchor=anchor,
                offset_left=offset_left,
                offset_top=offset_top,
            )
            for img in image
        ]

    from PIL import Image

    if fit is None or fit == "actual":
        left, top = 0, 0
        crop_left, crop_top = 0, 0
        image_width, image_height = image.size

        if anchor is not None:
            top_part, left_part = anchor.split("-")

            if top_part == "center":
                top = height // 2 - image_height // 2
            elif top_part == "bottom":
                top = height - image_height

            if left_part == "center":
                left = width // 2 - image_width // 2
            elif left_part == "right":
                left = width - image_width

        blank_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        if offset_top is not None:
            top += offset_top
        if offset_left is not None:
            left += offset_left
        if image.mode == "RGBA":
            blank_image.paste(image, (left, top), image)
        else:
            blank_image.paste(image, (left, top))

        return blank_image

    elif fit == "contain":
        image_width, image_height = image.size
        width_ratio, height_ratio = width / image_width, height / image_height
        horizontal_image_width = int(image_width * width_ratio)
        horizontal_image_height = int(image_height * width_ratio)
        vertical_image_width = int(image_width * height_ratio) 
        vertical_image_height = int(image_height * height_ratio)
        top, left = 0, 0
        direction = None
        if width >= horizontal_image_width and height >= horizontal_image_height:
            input_image = image.resize((horizontal_image_width, horizontal_image_height))
            if anchor is not None:
                top_part, _ = anchor.split("-")
                if top_part == "center":
                    top = height // 2 - horizontal_image_height // 2
                elif top_part == "bottom":
                    top = height - horizontal_image_height
        elif width >= vertical_image_width and height >= vertical_image_height:
            input_image = image.resize((vertical_image_width, vertical_image_height))
            if anchor is not None:
                _, left_part = anchor.split("-")
                if left_part == "center":
                    left = width // 2 - vertical_image_width // 2
                elif left_part == "right":
                    left = width - vertical_image_width

        if offset_top is not None:
            top += offset_top
        if offset_left is not None:
            left += offset_left

        blank_image = Image.new("RGBA", (width, height))
        if input_image.mode == "RGBA":
            blank_image.paste(input_image, (left, top), input_image)
        else:
            blank_image.paste(input_image, (left, top))

        return blank_image

    elif fit == "cover":
        image_width, image_height = image.size
        width_ratio, height_ratio = width / image_width, height / image_height
        horizontal_image_width = math.ceil(image_width * width_ratio)
        horizontal_image_height = math.ceil(image_height * width_ratio)
        vertical_image_width = math.ceil(image_width * height_ratio)
        vertical_image_height = math.ceil(image_height * height_ratio)
        top, left = 0, 0
        direction = None
        if width <= horizontal_image_width and height <= horizontal_image_height:
            input_image = image.resize((horizontal_image_width, horizontal_image_height))
            if anchor is not None:
                top_part, _ = anchor.split("-")
                if top_part == "center":
                    top = height // 2 - horizontal_image_height // 2
                elif top_part == "bottom":
                    top = height - horizontal_image_height
        elif width <= vertical_image_width and height <= vertical_image_height:
            input_image = image.resize((vertical_image_width, vertical_image_height))
            if anchor is not None:
                _, left_part = anchor.split("-")
                if left_part == "center":
                    left = width // 2 - vertical_image_width // 2
                elif left_part == "right":
                    left = width - vertical_image_width
        else:
            input_image = image.resize((width, height))  # We're probably off by a pixel

        if offset_top is not None:
            top += offset_top
        if offset_left is not None:
            left += offset_left

        blank_image = Image.new("RGBA", (width, height))
        if input_image.mode == "RGBA":
            blank_image.paste(input_image, (left, top), input_image)
        else:
            blank_image.paste(input_image, (left, top))

        return blank_image

    elif fit == "stretch":
        return image.resize((width, height)).convert("RGBA")

    else:
        raise ValueError(f"Unknown fit {fit}")

def scale_image(
    image: Image,
    scale: Union[int, float]=1.0,
    nearest: int=1,
    smallest: Optional[int]=None,
) -> Image:
    """
    Scales an image proportionally.
    """
    from PIL import Image
    image_width, image_height = image.size
    width = float(image_width*scale)
    height = float(image_height*scale)
    if smallest is not None:
       smallest_side = min(width, height)
       if smallest_side < smallest:
            additional_upscale = smallest / smallest_side
            width *= additional_upscale
            height *= additional_upscale
    image_width = nearest * round(width / nearest)
    image_height = nearest * round(height / nearest)
    return image.resize((image_width, image_height), Image.Resampling.LANCZOS)
