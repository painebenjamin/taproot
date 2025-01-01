from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING
from typing_extensions import Literal

from taproot.util import to_pil_array, is_multi_image
from taproot.constants import *
from taproot.tasks.base import Task

from .pretrained import PretrainedMidasDetector

if TYPE_CHECKING:
    from taproot.hinting import ImageType, ImageResultType
    from .inference import MiDaSInference # type: ignore[attr-defined]

__all__ = ["MiDaSDepthDetection"]

class MiDaSDepthDetection(Task):
    """
    MiDaS Depth Detection
    """

    """Global Task Metadata"""
    task = "depth-detection"
    model = "midas"
    default = True
    display_name = "MiDaS Depth Detection"
    pretrained_models = {"midas": PretrainedMidasDetector}
    static_memory_gb = .12464 # 124.64 MB, measured
    static_gpu_memory_gb = .25565 # 255.65 MB, measured

    """Author Metadata"""
    author = "René Ranftl"
    author_additional = ["Alexey Bochkovskiy", "Vladlen Koltun"]
    author_url = "https://arxiv.org/abs/2103.13413"
    author_journal = "arXiv"
    author_journal_volume = "2103.13413"
    author_journal_title = "Vision Transformers for Dense Prediction"
    author_journal_year = 2021

    """License Metadata"""
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
            "torchvision": TORCHVISION_VERSION_SPEC,
            "timm": ">=0.9",
            "einops": EINOPS_VERSION_SPEC,
            "scipy": SCIPY_VERSION_SPEC,
            "torchdiffeq": "~=0.2",
            "safetensors": SAFETENSORS_VERSION_SPEC,
        }

    """Internal Task Attributes"""

    @property
    def detector(self) -> MiDaSInference:
        """
        The MiDaS detector
        """
        return self.pretrained.midas

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        image: ImageType,
        mode: Literal["depth", "normal", "depth-normal"]="depth",
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL="png",
        output_upload: bool=False
    ) -> ImageResultType:
        """
        Perform MiDaS depth detection on the input image.

        :param image: The input image
        :param mode: The output mode. Choose between "depth", "normal", or "depth-normal"
        :param output_format: The format of the output image.
        :param output_upload: Whether to upload the output image to the configured storage backend and return the URL, or return the image data directly.
        :return: The depth or normal image
        """
        images = to_pil_array(image)
        resolution = max(max(i.size) for i in images)
        depth_and_normal = mode in ["normal", "depth-normal"]
        results = [
            self.detector.predict(
                i,
                output_type="pil",
                image_resolution=resolution,
                depth_and_normal=depth_and_normal,
            )
            for i in images
        ]
        if depth_and_normal:
            # Results are tuples of depth and normal
            if mode == "normal":
                # Discard depth
                results = [
                    n.resize(images[i].size)
                    for i, (d, n) in enumerate(results)
                ]
            else:
                # Keep both
                results = [
                    (
                        d.resize(images[i].size),
                        n.resize(images[i].size)
                    )
                    for i, (d, n) in enumerate(results)
                ]
        else:
            # Results are depth
            results = [
                d.resize(images[i].size)
                for i, d in enumerate(results)
            ]

        return self.get_output_from_image_result(
            results,
            output_format=output_format,
            output_upload=output_upload,
            return_first_item = not is_multi_image(image)
        )
