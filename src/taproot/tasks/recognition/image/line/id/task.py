from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

from typing import Union

from taproot.util import (
    is_multi_image,
    to_bchw_tensor,
    get_image_width_height,
    scale_tensor,
)
from taproot.constants import *
from taproot.tasks.base import Task

from .pretrained import (
    PretrainedLineartDetector,
    PretrainedAnimeLineartDetector,
    PretrainedCoarseLineartDetector,
)


if TYPE_CHECKING:
    import torch
    from taproot.hinting import ImageType, ImageResultType
    from .model import Generator, UNetGenerator

__all__ = [
    "InformativeDrawingsLineartDetection",
    "InformativeDrawingsCoarseLineartDetection",
    "InformativeDrawingsAnimeLineartDetection",
]

class InformativeDrawingsLineartDetection(Task):
    """
    Lineart detection
    """
    
    """Global Task Metadata"""
    task = "line-detection"
    model = "informative-drawings"
    default = True
    display_name = "Informative Drawings Line Art Detection"
    pretrained_models = {"detector": PretrainedLineartDetector}
    static_memory_gb = .04553 # 45.53 MB, measured
    static_gpu_memory_gb = .00858 # 8.5 MB, measured

    """Authorship Metadata"""
    author = "Caroline Chan"
    author_url = "https://arxiv.org/abs/2203.12691"
    author_additional = ["Fredo Durand", "Phillip Isola"]
    author_affiliations = ["Massachusetts Institute of Technology"]
    author_journal = "arXiv"
    author_journal_volume = "2203.12691"
    author_journal_year = 2022
    author_journal_title = "Informative Drawings: Learning to Generate Line Drawings that Convey Geometry and Semantics"

    """Licensing Metadata"""
    license = LICENSE_MIT

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

    """Internal Task Attributes"""

    @property
    def detector(self) -> Union[Generator, UNetGenerator]:
        """
        The lineart detector
        """
        return self.pretrained.detector # type: ignore[no-any-return]

    """Private Methods"""

    def _adjust_image_for_model(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adjusts an image for the model, base does nothing
        """
        return image

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        image: ImageType,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL="png",
        output_upload: bool=False
    ) -> ImageResultType:
        """
        Detects line art in an image.

        :param image: The input image
        :param output_format: The output image format
        :param output_upload: Whether to upload the output image to the configured storage backend and return the URL, or return the image data directly.
        :return: The output image
        """
        from torch import inference_mode
        images = to_bchw_tensor(
            image,
            resize=(512, 512),
            num_channels=3
        ).to(self.device, dtype=self.dtype)
        with inference_mode():
            results = self.detector(
                self._adjust_image_for_model(images)
            )
            results = scale_tensor(
                1.0-results,
                size=get_image_width_height(image)
            )
            return self.get_output_from_image_result(
                results,
                output_format=output_format,
                output_upload=output_upload,
                return_first_item=not is_multi_image(image)
            )

class InformativeDrawingsCoarseLineartDetection(InformativeDrawingsLineartDetection):
    """
    Coarse lineart detection
    """

    """Global Task Metadata"""
    model = "informative-drawings-coarse"
    default = False
    display_name = "Informative Drawings Coarse Line Art Detection"
    pretrained_models = {"detector": PretrainedCoarseLineartDetector}

class InformativeDrawingsAnimeLineartDetection(InformativeDrawingsLineartDetection):
    """
    Anime Lineart Detection Task
    """

    """Global Task Metadata"""
    model = "informative-drawings-anime"
    default = False
    display_name = "Informative Drawings Anime Line Art Detection"
    pretrained_models = {"detector": PretrainedAnimeLineartDetector}
    static_memory_gb = .04579 # 45.79 MB, measured
    static_gpu_memory_gb = .10881 # 108.81 MB, measured

    """Private Methods"""

    def _adjust_image_for_model(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adjusts an image for the model, base does nothing
        """
        return image * 2.0 - 1.0
