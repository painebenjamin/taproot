from __future__ import annotations

from typing import Dict, Optional, List, TYPE_CHECKING
from math import floor

from taproot.tasks.base import Task
from taproot.constants import *
from taproot.util import (
    floor_power,
    to_pil_array,
    is_multi_image,
    seed_all,
    SeedType
)

from .pretrained import (
    PretrainedAuraSR,
    PretrainedAuraSRV2
)

if TYPE_CHECKING:
    from taproot.hinting import ImageType, ImageResultType
    from PIL import Image
    from .model import AuraSR # type: ignore[attr-defined]

__all__ = [
    "AuraSuperResolution",
    "AuraSuperResolutionV2"
]

class AuraSuperResolution(Task):
    """
    Aura-SR from FAL
    """

    """Global Task Metadata"""
    task = "super-resolution"
    model = "aura"
    display_name = "Aura Super Resolution"
    default = False
    static_memory_gb = 0.09802
    static_gpu_memory_gb = 1.24

    """Internal Task Attributes"""
    upsampler: AuraSR
    pretrained_class = PretrainedAuraSR

    """Authorship Metadata"""
    author = "fal.ai"
    author_url = "https://blog.fal.ai/introducing-aurasr-an-open-reproduction-of-the-gigagan-upscaler-2/"
    author_journal = "fal.ai blog"
    author_journal_year = 2024
    author_journal_title = "Introducing AuraSR - An open reproduction of the GigaGAN Upscaler"

    """License Metadata"""
    license = LICENSE_CC_BY_SA_4

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
            "einops": EINOPS_VERSION_SPEC,
            "safetensors": SAFETENSORS_VERSION_SPEC,
        }

    @classmethod
    def required_files(cls, allow_optional: bool=True) -> List[str]:
        """
        Required files
        """
        return [cls.pretrained_class.model_url]

    """Internal properties for task"""

    @property
    def input_image_size(self) -> int:
        """
        The input image size
        """
        if not hasattr(self, "_input_image_size"):
            memory_gb = self.get_capability().gpus[self.gpu_index].memory_total / 1e9
            if memory_gb > 12:
                self._input_image_size = 128
            else:
                self._input_image_size = 64
        return self._input_image_size

    @property
    def batch_size(self) -> int:
        """
        The batch size
        """
        if not hasattr(self, "_batch_size"):
            memory_gb = self.get_capability().gpus[self.gpu_index].memory_total / 1e9
            if memory_gb > 16:
                self._batch_size = 16
            elif memory_gb > 8:
                self._batch_size = 8
            else:
                self._batch_size = 4
        return self._batch_size

    """Internal Methods"""

    def get_batch_size(self, pixel_count: int) -> int:
        """
        The batch size for this task
        """
        image_size = pixel_count * 1024
        vram_free = self.get_capability().gpus[self.gpu_index].memory_free
        theoretical_batch_size = floor_power(vram_free / image_size)
        return max(1, min(128, theoretical_batch_size))

    def upsample(self, image: Image.Image, amount: float=4.0) -> Image.Image:
        """
        Upsample the image
        """
        width, height = image.size
        target_width = floor(width * amount)
        target_height = floor(height * amount)

        while amount > 1.0:
            image = self.upscaler.upscale_4x(
                image,
                max_batch_size=self.batch_size,
                dtype=self.dtype
            )
            amount /= 4.0

        if amount != 1.0:
            image = image.resize((target_width, target_height))

        return image

    """Overrides"""

    def load(self) -> None:
        """
        Load the model
        """
        self.upscaler = self.pretrained_class.instantiate_and_load_from_url_to_dir(
            self.model_dir,
            device=self.device,
            dtype=self.dtype,
            image_size=self.input_image_size*4,
            input_image_size=self.input_image_size,
        )

    def __call__( # type: ignore[override]
        self,
        *,
        image: ImageType,
        amount: float=4.0,
        seed: SeedType=None,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL="png",
        output_upload: bool=False,
    ) -> ImageResultType:
        """
        Upscales an image, increasing its resolution by a factor using the Aura Super Resolution model.

        :param image: The image to upscale.
        :param amount: The amount to upscale the image by.
        :param seed: The seed to use for random operations.
        :param output_format: The format of the output image.
        :param output_upload: Whether to upload the output image to the configured storage backend and return the URL, or return the image data directly.
        :return: The upscaled image.
        """
        self.save_format = "jpg"
        seed_all(seed)
        images = to_pil_array(
            image,
            num_channels=3,
            directory=self.save_dir
        )
        results = [
            self.upsample(i, amount=amount)
            for i in images
        ]
        return self.get_output_from_image_result(
            results,
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=not is_multi_image(image)
        )

class AuraSuperResolutionV2(AuraSuperResolution):
    """
    Aura-SR from FAL, Version 2
    """

    """Global Task Metadata"""
    model = "aura-v2"
    default = True
    display_name = "Aura Super Resolution V2"

    """Internal Task Attributes"""
    pretrained_class = PretrainedAuraSRV2

    """Authorship Metadata"""
    author_url = "https://blog.fal.ai/aurasr-v2/"
    author_journal_title = "AuraSR V2"
