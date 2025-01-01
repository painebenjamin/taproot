from __future__ import annotations

from typing import Dict, Optional, List, TYPE_CHECKING

from taproot.util import (
    to_bchw_tensor,
    is_multi_image,
    get_image_width_height
)
from taproot.constants import *
from taproot.tasks.base import Task

from .pretrained import PretrainedMobileLineSegmentDetector
from .util import pred_lines, draw_lines

if TYPE_CHECKING:
    from PIL.Image import Image
    from taproot.hinting import ImageType, ImageResultType
    from .model import MLSD

__all__ = ["MobileLineSegmentDetection"]

class MobileLineSegmentDetection(Task):
    """
    Mobile Line Segment Detection (MLSD) task
    """

    """Global Task Metadata"""
    task = "line-detection"
    model = "mlsd"
    default = False
    display_name = "Mobile Line Segment Detection"
    pretrained_models = {"detector": PretrainedMobileLineSegmentDetector}
    static_memory_gb = .04994 # 49.94 MB, measured
    static_gpu_memory_gb = .00322 # 3.22 MB, measured

    """Authorship Metadata"""
    author = "Geonmo Gu"
    author_url = "https://arxiv.org/abs/2106.00186"
    author_additional = ["Byungsoo Ko", "SeongHyun Go", "Sung-Hyun Lee", "Jingeun Lee", "Minchul Shin"]
    author_affiliations = ["NAVER/LINE Vision"]
    author_journal = "arXiv"
    author_journal_year = 2022
    author_journal_volume = "2106.00186"
    author_journal_title = "Towards Light-weight and Real-time Line Segment Detection"

    """Licensing Metadata"""
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
            "safetensors": SAFETENSORS_VERSION_SPEC,
        }

    """Local Task Attributes"""

    @property
    def detector(self) -> MLSD:
        """
        The line segment detector
        """
        return self.pretrained.detector # type: ignore[no-any-return]

    def __call__( # type: ignore[override]
        self,
        *,
        image: ImageType,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL="png",
        output_upload: bool=False,
    ) -> ImageResultType:
        """
        Detect line segments in an image.

        :param image: The input image
        :param output_format: The output image format
        :param output_upload: Whether to upload the output image to the configured storage backend and return the URL, or return the image data directly.
        :return: The output image
        """
        import torch
        with torch.inference_mode():
            images = to_bchw_tensor(
                image,
                dtype=self.dtype,
                resize=(512,512)
            ).to(self.device)
            b, c, h, w = images.shape
            images = torch.cat([
                images,
                torch.ones(
                    (b, 1, h, w),
                    device=self.device,
                    dtype=self.dtype
                )
            ], dim=1)
            outputs = self.detector(images * 2.0 - 1.0)

        width, height = get_image_width_height(image)
        drawn_images: List[Image] = []
        for output in outputs:
            lines = pred_lines(output.unsqueeze(0))
            drawn_images.append(draw_lines(width, height, lines))

        return self.get_output_from_image_result(
            drawn_images,
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=not is_multi_image(image)
        )
