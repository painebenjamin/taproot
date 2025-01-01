from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING
from typing_extensions import Literal

from taproot.util import to_pil_array, is_multi_image
from taproot.constants import *
from taproot.tasks.base import Task

from .pretrained import PretrainedBackgroundRemover

if TYPE_CHECKING:
    from PIL import Image
    from taproot.hinting import ImageType, ImageResultType
    from .u2net import U2NET # type: ignore[attr-defined]

__all__ = ["BackgroundRemovalBackgroundRemover"]

class BackgroundRemovalBackgroundRemover(Task):
    """
    Background removal using BackgroundRemover
    """

    """Global Task Metadata"""
    task = "background-removal"
    model = "backgroundremover"
    default = True
    display_name = "BackgroundRemover"
    pretrained_models = {"net": PretrainedBackgroundRemover}
    static_memory_gb = .04579 # 45.79 MB, measured
    static_gpu_memory_gb = .21762 # 217.62 MB, measured

    """Authorship metadata"""
    author = "Johnathan Nader"
    author_additional = ["Lucas Nestler", "Dr. Tim Scarfe", "Daniel Gatis"]
    author_url = "https://github.com/nadermx/backgroundremover"

    """License metadata"""
    license = LICENSE_APACHE

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
            "skimage": SKIMAGE_VERSION_SPEC,
            "scipy": SCIPY_VERSION_SPEC,
            "safetensors": SAFETENSORS_VERSION_SPEC,
        }

    """Internal Task Attributes"""

    @property
    def net(self) -> U2NET:
        """
        A proxy to the BackgroundRemover model.
        """
        return self.pretrained.net

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        image: ImageType,
        mode: Literal["mask", "composite"]="composite",
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL="png",
        output_upload: bool = False,
    ) -> ImageResultType:
        """
        Removes background from one or more images using BackgroundRemover.

        :param image: The image or images to remove the background from.
        :param mode: The output mode. If "mask", returns the mask only. If "composite", returns the image with the mask applied.
        :param output_format: The output format for the image.
        :param output_upload: Whether to upload the output image to the configured storage backend and return the URL, or return the image data directly.
        :return: The image with the background removed.
        """
        from PIL import Image
        from .detect import predict # type: ignore[attr-defined]
        images = to_pil_array(image)
        masks = predict(self.net, images, self.device)

        if mode == "mask":
            results = [
                m.resize(i.size)
                for i, m in zip(images, masks)
            ]
        else:
            results = [
                Image.composite(
                    i,
                    Image.new("RGBA", i.size, (0, 0, 0, 0)),
                    m.resize(i.size)
                )
                for i, m in zip(images, masks)
            ]

        return self.get_output_from_image_result(
            results,
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=not is_multi_image(image)
        )
