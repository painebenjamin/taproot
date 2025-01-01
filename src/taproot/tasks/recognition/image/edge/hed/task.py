from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

from taproot.util import to_pil_array, is_multi_image
from taproot.constants import *
from taproot.tasks.base import Task

from .pretrained import PretrainedHEDDetector

if TYPE_CHECKING:
    from taproot.hinting import ImageType, ImageResultType
    from .model import HEDDetector # type: ignore[attr-defined]

__all__ = ["HolisticallyNestedEdgeDetection"]

class HolisticallyNestedEdgeDetection(Task):
    """
    Holistically-Nested Edge Detection
    """

    """Global Task Metadata"""
    task = "edge-detection"
    model = "hed"
    default = False
    display_name = "Holistically-Nested Edge Detection"
    pretrained_models = {"detector": PretrainedHEDDetector}
    static_memory_gb = .04553
    static_gpu_memory_gb = .02944

    """Author Metadata"""
    author = "Saining Xie"
    author_url = "https://arxiv.org/abs/1504.06375"
    author_additional = ["Zhuowen Tu"]
    author_affiliations = ["University of California, San Diego"]
    author_journal = "arXiv"
    author_journal_volume = "1504.06375"
    author_journal_year = 2015
    author_journal_title = "Holistically-Nested Edge Detection"

    """License Metadata"""
    license = LICENSE_APACHE

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Required packages
        """
        return {
            "cv2": OPENCV_VERSION_SPEC,
            "pil": PILLOW_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
            "torch": TORCH_VERSION_SPEC,
            "einops": EINOPS_VERSION_SPEC,
            "safetensors": SAFETENSORS_VERSION_SPEC
        }

    """Internal Task Attributes"""

    @property
    def detector(self) -> HEDDetector:
        """
        The detector
        """
        return self.pretrained.detector

    def __call__( # type: ignore[override]
        self,
        *,
        image: ImageType,
        scribble: bool=False,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL="png",
        output_upload: bool=False
    ) -> ImageResultType:
        """
        Detect edges in an image.

        :param image: The input image to detect edges in.
        :param scribble: Whether to use scribble-format refinement.
        :param output_format: The format to return the output image in.
        :param output_upload: Whether to upload the output image to the configured storage backend and return the URL, or return the image data directly.
        :return: The detected edges in the image.
        """
        images = to_pil_array(image)
        resolution = min(min(i.size) for i in images)
        results = [
            self.detector(
                i,
                detect_resolution=resolution,
                image_resolution=resolution,
                scribble=scribble
            ).resize(i.size)
            for i in images
        ]
        return self.get_output_from_image_result(
            results,
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=not is_multi_image(image)
        )
