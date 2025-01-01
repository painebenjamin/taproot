from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

from taproot.util import to_pil_array, is_multi_image
from taproot.constants import *
from taproot.tasks.base import Task

if TYPE_CHECKING:
    from taproot.hinting import ImageType, ImageResultType

__all__ = ["CannyEdgeDetection"]

class CannyEdgeDetection(Task):
    """
    Canny Edge Detection
    """
    """Global task metadata"""
    task = "edge-detection"
    model = "canny"
    default = True
    display_name = "Canny Edge Detection"

    """Authorship metadata"""
    author = "John Canny"
    author_url = "https://ieeexplore.ieee.org/document/4767851"
    author_journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence"
    author_journal_year = 1986
    author_journal_volume = "6"
    author_journal_pages = "679-698"
    author_journal_title = "A Computational Approach to Edge Detection"
    implementation_author = "OpenCV"
    implementation_author_url = "https://opencv.org/"

    """License metadata"""
    license = LICENSE_APACHE

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Required packages
        """
        return {
            "cv2": OPENCV_VERSION_SPEC,
            "pil": PILLOW_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC
        }

    def __call__( # type: ignore[override]
        self,
        *,
        image: ImageType,
        lower_bound: int=100,
        upper_bound: int=200,
        output_channels: int=3,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL="png",
        output_upload: bool=False,
    ) -> ImageResultType:
        """
        Perform Canny edge detection on the input image(s).

        :param image: Input image(s) to process.
        :param lower_bound: Lower bound for the Canny edge detection algorithm.
        :param upper_bound: Upper bound for the Canny edge detection algorithm.
        :param output_channels: Number of output channels, use 3 for RGB images or 1 for grayscale images.
        :param output_format: Output format for the processed image(s).
        :param output_upload: Whether to upload the output image to the configured storage backend and return the URL, or return the image data directly.
        :return: Processed image(s).
        """
        import cv2 # type: ignore[import-not-found,unused-ignore]
        import numpy as np
        from PIL import Image
        input_images = to_pil_array(image)
        canny_array = [
            cv2.Canny(
                np.array(input_image),
                lower_bound,
                upper_bound
            )[:, :, None]
            for input_image in input_images
        ]
        canny_images = [
            Image.fromarray(
                np.concatenate([canny] * output_channels, axis=2)
            )
            for canny in canny_array
        ]
        return self.get_output_from_image_result(
            canny_images,
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=not is_multi_image(image)
        )
