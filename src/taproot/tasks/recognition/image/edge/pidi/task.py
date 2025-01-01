from __future__ import annotations

from typing import Dict, Optional, List, TYPE_CHECKING

from taproot.util import to_pil_array, is_multi_image
from taproot.constants import *
from taproot.tasks.base import Task

from .pretrained import PretrainedPidiDetector

if TYPE_CHECKING:
    from taproot.hinting import ImageType, ImageResultType
    from .net import PidiNetDetector # type: ignore[attr-defined]

__all__ = ["SoftEdgeDetection"]

class SoftEdgeDetection(Task):
    """
    Soft Edge (PIDI) Detection
    """

    """Global Task Metadata"""
    task = "edge-detection"
    model = "pidi"
    default = False
    display_name = "Soft Edge (PIDI) Detection"
    pretrained_models = {"detector": PretrainedPidiDetector}
    static_memory_gb = .04553 # 45 MB, measured
    static_gpu_memory_gb = .00140 # 1.4 MB, measured

    """Authorship Metadata"""
    author = "Zhuo Su"
    author_additional = ["Wenzhe Liu", "Zitong Yu", "Dewen Hu", "Qing Liao", "Qi Tian", "Matti Pietikäinen", "Li Liu"]
    author_journal = "Proceedings of the IEEE/CVF International Conference on Computer Vision"
    author_journal_year = 2021
    author_journal_pages = "5117-5127"
    author_journal_title = "Pixel Difference Networks for Efficient Edge Detection"

    """Licensing Metadata"""
    license = "MIT License with Non-Commercial Clause"
    license_url = "https://github.com/hellozhuo/pidinet/blob/master/LICENSE"
    license_commercial = False # Special clause from authors
    license_derivatives = True
    license_redistribution = True
    license_hosting = True
    license_copyleft = False

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
            "safetensors": SAFETENSORS_VERSION_SPEC,
        }

    @classmethod
    def required_files(cls, allow_optional: bool=True) -> List[str]:
        """
        Required files
        """
        from .pretrained import PretrainedPidiDetector
        return [PretrainedPidiDetector.model_url]

    """Internal Task Attributes"""

    @property
    def detector(self) -> PidiNetDetector:
        """
        The PIDI model
        """
        return self.pretrained.detector

    def __call__( # type: ignore[override]
        self,
        *,
        image: ImageType,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL="png",
        output_upload: bool=False
    ) -> ImageResultType:
        """
        Perform soft edge detection on the input image.

        :param image: The input image to process.
        :param output_format: The output format for the image.
        :param output_upload: Whether to upload the output image to the configured storage backend and return the URL, or return the image data directly.
        :return: The processed image.
        """
        images = to_pil_array(image, num_channels=3)
        resolution = min(min(i.size) for i in images)
        results = [
            self.detector(
                i,
                detect_resolution=resolution,
                image_resolution=resolution,
                safe=True,
            ).resize(i.size)
            for i in images
        ]
        return self.get_output_from_image_result(
            results,
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=not is_multi_image(image)
        )
