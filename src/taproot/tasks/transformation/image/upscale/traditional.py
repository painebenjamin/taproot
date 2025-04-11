from __future__ import annotations

import os

from typing import Dict, Optional, TYPE_CHECKING
from typing_extensions import Literal

from concurrent.futures import ThreadPoolExecutor

from taproot.constants import *
from taproot.tasks.base import Task
from taproot.util import (
    to_pil_array,
    is_multi_image,
)

if TYPE_CHECKING:
    from PIL import Image
    from taproot.hinting import ImageType, ImageResultType

RESAMPLE_METHOD_LITERAL = Literal["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"]

class BasicSuperResolution(Task):
    """
    Uses tradition (non-AI) methods to upscale images.
    """
    task = "super-resolution"
    model = None
    default = True
    display_name = "Traditional Super Resolution"

    """Authorship Metadata"""
    author = "Benjamin Paine"
    author_url = "https://github.com/painebenjamin/taproot"
    author_affiliations = ["Taproot"]
    implementation_author = "Pillow"
    implementation_author_url = "https://pillow.readthedocs.io/en/stable/"

    """Licensing Metadata"""
    license = LICENSE_APACHE

    """Method Overrides"""
    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Returns the required packages for this task.
        """
        return {
            "pil": PILLOW_VERSION_SPEC
        }

    """Private Methods"""

    def _get_resample_enum_by_name(
        self,
        method: RESAMPLE_METHOD_LITERAL
    ) -> Literal[0, 1, 2, 3, 4, 5]:
        """
        Get the resample enum by name.

        :param method: The resampling method to use. Options are "nearest", "box", "bilinear", "hamming", "bicubic", "lanczos".
        :return: The resampling enum.
        """
        from PIL import Image
        if method == "nearest":
            return Image.NEAREST # type: ignore[no-any-return,attr-defined]
        elif method == "box":
            return Image.BOX # type: ignore[no-any-return,attr-defined]
        elif method == "bilinear":
            return Image.BILINEAR # type: ignore[no-any-return,attr-defined]
        elif method == "hamming":
            return Image.HAMMING # type: ignore[no-any-return,attr-defined]
        elif method == "bicubic":
            return Image.BICUBIC # type: ignore[no-any-return,attr-defined]
        elif method == "lanczos":
            return Image.LANCZOS # type: ignore[no-any-return,attr-defined]
        else:
            raise ValueError(f"Unknown resampling method: {method}")

    def _upscale_image(
        self,
        image: Image.Image,
        amount: float=2.0,
        method: RESAMPLE_METHOD_LITERAL="lanczos"
    ) -> Image.Image:
        """
        Upscale an image using traditional methods.

        :param image: The input image to upscale.
        :param amount: The amount to upscale the image by.
        :param method: The resampling method to use. Options are "nearest", "box", "bilinear", "hamming", "bicubic", "lanczos".
        :return: The upscaled image.
        """
        width, height = image.size
        new_width = int(width * amount)
        new_height = int(height * amount)
        return image.resize(
            (new_width, new_height),
            resample=self._get_resample_enum_by_name(method)
        )

    """Method Implementation"""
    def __call__( # type: ignore[override]
        self,
        *,
        image: ImageType,
        method: RESAMPLE_METHOD_LITERAL="bicubic",
        amount: float=2.0,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL="png",
        output_upload: bool=False
    ) -> ImageResultType:
        """
        Upscale an image using traditional methods.

        :param image: The input image(s) to upscale.
        :param method: The resampling method to use. Options are "nearest", "box", "bilinear", "hamming", "bicubic", "lanczos".
        :param amount: The amount to upscale the image by.
        :param output_format: The format to output the image as.
        :param output_upload: Whether to upload the image(s) to the cloud.
        :return: The upscaled image(s).
        """
        input_images = to_pil_array(image)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            output_futures = []
            for input_image in input_images:
                output_future = executor.submit(self._upscale_image, input_image, amount, method)
                output_futures.append(output_future)

            results = [
                output_future.result()
                for output_future
                in output_futures
            ]
            return self.get_output_from_image_result(
                results,
                output_format=output_format,
                output_upload=output_upload,
                return_first_item=not is_multi_image(image)
            )
