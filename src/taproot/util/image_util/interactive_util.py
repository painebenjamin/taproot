from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .normalization_util import to_pil_array

if TYPE_CHECKING:
    from taproot.hinting import ImageType

__all__ = [
    "show_image",
]

def show_image(
    image: ImageType,
    title: Optional[str]=None,
) -> None:
    """
    Shows an image
    """
    image = to_pil_array(image)[0]
    try:
        from IPython.display import display # type: ignore[import-not-found]
        display(image)
    except ImportError:
        import cv2
        import numpy as np
        # Show until a key is pressed
        cv2.imshow(title or "Image", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
