from __future__ import annotations

from typing import Dict, Optional, List, TYPE_CHECKING
from typing_extensions import Literal

from taproot.util import to_pil_array, is_multi_image
from taproot.constants import *
from taproot.tasks.base import Task

if TYPE_CHECKING:
    from taproot.hinting import ImageType, ImageResultType
    from .helper import DWPoseDetector # type: ignore[attr-defined]

__all__ = ["DWPoseDetection"]

class DWPoseDetection(Task):
    """
    DWPose detection and masking
    """

    """Global Task Metadata"""
    task = "pose-detection"
    model = "dwpose"
    display_name = "DWPose Pose Detection"
    default = True
    static_memory_gb = .18724 # 187.24 MB, measured
    static_gpu_memory_gb = .35464 # 354.64 MB, measured

    """Authorship Metadata"""
    author = "Zhengdong Yang"
    author_url = "https://arxiv.org/abs/2307.15880"
    author_additional = ["Ailing Zeng", "Chun Yuan", "Yu Li"]
    author_affiliations = ["Tsinghua Zhenzhen International Graduate School", "International Digital Economy Academy (IDEA)"]
    author_journal = "arXiv"
    author_journal_volume = "2307.15880"
    author_journal_year = 2023
    author_journal_title = "Effective Whole-body Pose Estimation with Two-stages Distillation"

    """Licensing Metadata"""
    license = LICENSE_APACHE

    """Task-Specific Metadata"""
    estimation_model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/pose-detection-dwpose-estimation.safetensors"
    detection_model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/pose-detection-dwpose-detection.safetensors"
    detector: DWPoseDetector

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Required packages
        """
        return {
            "transformers": TRANSFORMERS_VERSION_SPEC,
            "safetensors": SAFETENSORS_VERSION_SPEC,
            "accelerate": ACCELERATE_VERSION_SPEC,
            "pil": PILLOW_VERSION_SPEC,
            "torch": TORCH_VERSION_SPEC,
            "torchvision": TORCHVISION_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
            "scipy": SCIPY_VERSION_SPEC,
            "skimage": SKIMAGE_VERSION_SPEC,
            "sklearn": SKLEARN_VERSION_SPEC,
            "cv2": OPENCV_VERSION_SPEC,
            "ninja": "~=1.11.1",
            "openmim": "~=0.3.9",
            "mmcv": ">=2.0.1,<2.2",
            "mmdet": ">=3.0.0,<3.3",
            "mmpose": ">=1.1.0",
            "mmengine": ">=0.10.0",
            "ordered_set": "~=4.0",
        }

    @classmethod
    def required_files(cls, allow_optional: bool=True) -> List[str]:
        """
        Required files
        """
        return [cls.estimation_model_url, cls.detection_model_url]

    """Internal properties for task"""

    @property
    def estimation_model_file(self) -> str:
        """
        Estimation model file
        """
        return self.get_model_file(self.estimation_model_url)

    @property
    def detection_model_file(self) -> str:
        """
        Detection model file
        """
        return self.get_model_file(self.detection_model_url)

    """Override the load method to load the model"""

    def load(self) -> None:
        """
        Load the model
        """
        from .helper import DWPoseDetector # type: ignore[attr-defined]
        self.detector = DWPoseDetector(
            det_ckpt=self.detection_model_file,
            pose_ckpt=self.estimation_model_file,
        )
        self.detector.to(self.device, dtype=self.dtype)

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
            return_first_item=not is_multi_image(image)
        )
