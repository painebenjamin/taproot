from __future__ import annotations

# All utility methods are available in the `taproot.util` module and
# can be imported as needed. They will all lazy import the necessary
# dependencies when first called, so you can import them all at once
# without worrying about the overhead of importing unused modules.
from taproot.util import is_multi_image, to_bchw_tensor
from taproot.constants import *
from taproot.tasks.base import Task

from .pretrained import PretrainedTemplateModel

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    # Unlike the utility methods, the model classes must be imported
    # in the `TYPE_CHECKING` block to avoid unnecessarily initializing
    # libraries at runtime.
    from taproot.hinting import ImageType, ImageResultType
    from .model import TemplateModel

__all__ = ["TaskTemplate"]

class TaskTemplate(Task):
    """
    Task template example.

    Lets all the defaults do their work - so required packages, files,
    loading, unloading, etc. is all configured via setting `pretrained_models`
    to your model class.
    """

    """Global Task Metadata"""

    # The tuple of task and model must be unique
    task: str = "task-name"
    model: Optional[str] = None
    # Only the first task found with default=True will be used as the default
    default: bool = True
    static_gpu_memory_gb: Optional[float] = 0.000128 # 128 KB
    static_memory_gb: Optional[float] = 0.04874 # 48.74 MB (torch)
    # Optional, but generally the easiest way to configure models for inference
    pretrained_models = {
        "model": PretrainedTemplateModel
    }

    """Internal Task Attributes"""

    @property
    def module(self) -> TemplateModel:
        """
        Add mapped modules as properties for convenience and type hinting.
        """
        return self.pretrained.model # type: ignore[no-any-return]

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        image: ImageType,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL = "png",
        output_upload: bool = False,
    ) -> ImageResultType:
        """
        Invoke the template task.
        """
        import torch
        with torch.inference_mode():
            # Use utility methods to standardize the input
            images = to_bchw_tensor(
                image,
                dtype=self.dtype,
                resize=None, # (width, height)
            ).to(self.device)
            # images is a fp32/fp16 tensor in the range [0, 1]
            # Do your work here
            results = self.module(images)

        # This utility method will get the requested format
        return self.get_output_from_image_result(
            results,
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=not is_multi_image(image)
        )
