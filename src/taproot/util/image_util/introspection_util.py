from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

from ..introspection_util import is_numpy_array, is_torch_tensor, is_pil_image

if TYPE_CHECKING:
    from ...hinting import ImageType

__all__ = [
    "is_multi_image",
    "get_image_width_height",
]

def is_multi_image(images: ImageType) -> bool:
    """
    Determines if an image was intended to be a multi-frame image
    This is from the perspective of the developer, so generally this
    should check if the image is:
        1. A list of images
        2. A 4-dimensional tensor
        3. A 4-dimensional numpy array

    Notably we do NOT check if it has more than one frame, just that the
    developer passed it in a container that COULD have more than one.

    Others will be treated as singular images, so functions should return
    a singular image as well.
    """
    if is_numpy_array(images):
        return len(images.shape) == 4
    elif is_torch_tensor(images):
        return images.ndimension() == 4
    return isinstance(images, list)

def get_image_width_height(image: ImageType) -> Tuple[int, int]:
    """
    Gets the size of an image
    """
    if isinstance(image, list):
        return get_image_width_height(image[0])
    elif is_pil_image(image):
        return image.size
    elif is_numpy_array(image):
        height, width = image.shape[:2]
        return width, height
    elif is_torch_tensor(image):
        height, width = image.shape[-2:]
        return width, height
    raise ValueError(f"Unsupported image type: {type(image).__name__}")
