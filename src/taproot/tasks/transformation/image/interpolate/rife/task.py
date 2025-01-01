from __future__ import annotations

from typing import Optional, Dict, TYPE_CHECKING

from taproot.util import to_bchw_tensor, pad_image
from taproot.constants import *
from taproot.tasks.base import Task

from .pretrained import RIFENetModel

if TYPE_CHECKING:
    import torch
    from taproot.hinting import ImageType, ImageResultType
    from .model import IFNet

__all__ = [
    "RIFEImageInterpolation",
]

class RIFEImageInterpolation(Task):
    """
    Frame Interpolation with RIFE (Real-Time Intermediate Flow Estimation)
    """

    """Global Task Metadata"""

    # The tuple of task and model must be unique
    task = "image-interpolation"
    model = "rife"
    default = False
    display_name = "Real-Time Intermediate Flow Estimation (RIFE) Image Interpolation"
    pretrained_models = {"model": RIFENetModel}
    static_memory_gb = 0.04879
    static_gpu_memory_gb = 0.02268

    """Author metadata"""
    author = "Zhewei Huang"
    author_additional = ["Tianyuan Zhang", "Wen Heng", "Boxin Shi", "Shuchang Zhou"]
    author_affiliations = [
        "Megvii Research",
        "NERCVT, School of Computer Science, Peking University",
        "Institute for Artificial Intelligence, Peking University",
        "Beijing Academy of Artificial Intelligence"
    ]
    author_url = "https://arxiv.org/abs/2011.06294"
    author_journal = "ECCV"
    author_journal_year = 2022
    author_journal_title = "Real-Time Intermediate Flow Estimation for Video Frame Interpolation"

    """License metadata"""
    license = LICENSE_MIT

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Required packages.
        """
        return {
            "pil": PILLOW_VERSION_SPEC,
            "torch": TORCH_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
            "safetensors": SAFETENSORS_VERSION_SPEC,
        }

    """Internal Task Attributes"""

    @property
    def module(self) -> IFNet:
        """
        Add mapped modules as properties for convenience and type hinting.
        """
        return self.pretrained.model # type: ignore[no-any-return]

    def interpolate_frames(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        num_frames: int = 1
    ) -> torch.Tensor:
        """
        Runs the frame interpolation network, returning all frames including
        the start and end frames.
        """
        import torch
        assert start.shape == end.shape, "Start and end frames must have the same shape"
        b, c, h, w = start.shape
        pad_w = ((w - 1) // 128 + 1) * 128
        pad_h = ((h - 1) // 128 + 1) * 128
        padding = (0, pad_w - w, 0, pad_h - h)

        start = pad_image(start, padding) # type: ignore[assignment]
        end = pad_image(end, padding) # type: ignore[assignment]

        timesteps = torch.linspace(0, 1, num_frames + 2)[1:-1]
        timesteps = timesteps.to(self.device, dtype=self.dtype)

        middle = [
            self.module(
                torch.cat([start, end], dim=1),
                timestep=t,
            )
            for t in timesteps
        ]

        return torch.cat([start] + middle + [end], dim=0)

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        start: ImageType,
        end: ImageType,
        num_frames: int = 1,
        include_ends: bool = False,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL = "png",
        output_upload: bool = False,
    ) -> ImageResultType:
        """
        Interpolates frames between two images using RIFE (Real-Time Intermediate Flow Estimation).

        :param start: The starting image.
        :param end: The ending image.
        :param num_frames: The number of frames to interpolate between the start and end images.
        :param include_ends: Whether to include the start and end images in the output.
        :param output_format: The format of the output image.
        :param output_upload: Whether to upload the output image to the configured storage backend and return the URL, or return the image data directly.
        """
        import torch
        with torch.inference_mode():
            # Use utility methods to standardize the input
            start = to_bchw_tensor(start, num_channels=3, dtype=self.dtype).to(self.device)
            end = to_bchw_tensor(end, num_channels=3, dtype=self.dtype).to(self.device)

            results = self.interpolate_frames(
                start,
                end,
                num_frames=num_frames
            )
            if not include_ends:
                results = results[1:-1]

        return self.get_output_from_image_result(
            results,
            output_format=output_format,
            output_upload=output_upload,
            return_first_item=False
        )
