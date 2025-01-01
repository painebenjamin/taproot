from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING
from typing_extensions import Literal

from taproot.util import to_pil_array, is_multi_image
from taproot.constants import *
from taproot.tasks.base import Task

from .pretrained import PretrainedOpenposeDetector

if TYPE_CHECKING:
    from taproot.hinting import ImageType, ImageResultType
    from .helper import OpenposeDetector # type: ignore[attr-defined]

__all__ = ["OpenPoseDetection"]

class OpenPoseDetection(Task):
    """
    OpenPose pose detection and masking
    """

    """Global Task Metadata"""
    task = "pose-detection"
    model = "openpose"
    default = False
    display_name = "OpenPose Pose Detection"
    pretrained_models = {"detector": PretrainedOpenposeDetector}
    static_memory_gb = .1260 # 126 MB, measured
    static_gpu_memory_gb = .25996 # 260 MB, measured

    """Authorship Metadata"""
    author = "Zhe Cao"
    author_additional = ["Gines Hidalgo", "Tomas Simon", "Shih-En Wei", "Yaser Sheikh"]
    author_url = "https://arxiv.org/abs/1812.08008"
    author_journal = "arXiv"
    author_journal_volume = "1812.08008"
    author_journal_year = 2018
    author_journal_title = "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields"

    """Licensing Metadata"""
    license = "OpenPose Academic or Non-Profit Non-Commercial Research License"
    license_url = "https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE"
    license_attribution = True
    license_derivatives = True
    license_redistribution = True
    license_copy_left = False
    license_commercial = False
    license_hosting = False

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Required packages
        """
        return {
            "pil": PILLOW_VERSION_SPEC,
            "torch": TORCH_VERSION_SPEC,
            "torchvision": TORCHVISION_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
            "scipy": SCIPY_VERSION_SPEC,
            "skimage": SKIMAGE_VERSION_SPEC,
            "cv2": OPENCV_VERSION_SPEC,
            "einops": EINOPS_VERSION_SPEC,
            "matplotlib": "~=3.8",
            "safetensors": SAFETENSORS_VERSION_SPEC
        }

    """Internal Task Metadata"""

    @property
    def detector(self) -> OpenposeDetector:
        """
        The OpenPose detector
        """
        return self.pretrained.detector

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        image: ImageType,
        mode: Literal["pose", "mask"]="pose",
        hands: bool=True,
        face: bool=True,
        body: bool=True,
        composite: bool=True,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL="png",
        output_upload: bool=False
    ) -> ImageResultType:
        """
        Detect poses in the input image(s).

        :param image: Input image(s) to process.
        :param mode: Detection mode, either "pose" or "mask".
        :param hands: Whether to detect hands.
        :param face: Whether to detect faces.
        :param body: Whether to detect the body.
        :param composite: Whether to composite the output image(s) or return the isolated poses.
        :param output_format: Output format for the processed image(s).
        :param output_upload: Whether to upload the output image to the configured storage backend and return the URL, or return the image data directly.
        :return: Processed image(s).
        """
        draw_type = "pose"
        if mode == "mask":
            body = False
            if hands and face:
                draw_type = "mask"
            elif hands:
                draw_type = "handmask"
            elif face:
                draw_type = "facemask"
            else:
                raise ValueError("Either hands or face must be True when using mask mode.")

        images = to_pil_array(image)
        results = [
            self.detector(
                i,
                include_body=body,
                include_hand=hands,
                include_face=face,
                draw_type=draw_type,
                isolated=not composite
            )
            for i in images
        ]

        for i, (source_image, result) in enumerate(zip(images, results)):
            if not composite:
                for j, result_image in enumerate(result):
                    result[j] = result_image.resize(source_image.size)
            else:
                results[i] = result.resize(source_image.size)

        return self.get_output_from_image_result(
            results,
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=not is_multi_image(image) and composite
        )
