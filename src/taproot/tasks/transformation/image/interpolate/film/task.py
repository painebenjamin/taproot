from __future__ import annotations

from typing import Optional, Dict, TYPE_CHECKING

from taproot.util import to_bchw_tensor, pad_image_to_nearest
from taproot.constants import *
from taproot.tasks.base import Task

from .pretrained import FILMNetModel

if TYPE_CHECKING:
    import torch
    from taproot.hinting import ImageType, ImageResultType

__all__ = ["FILMInterpolation"]

class FILMInterpolation(Task):
    """
    Frame Interpolation with FiLM (Frame Interpolation for Large Motion)
    """

    """Global Task Metadata"""
    task = "image-interpolation"
    model = "film"
    default = True
    display_name = "Frame Interpolation for Large Motion (FiLM) Image Interpolation"
    pretrained_models = {"model": FILMNetModel}
    static_memory_gb = 0.04862 # 48.62 MB
    static_gpu_memory_gb = 0.07 # 70 MB

    """Authorship metadata"""
    author: str = "Fitsum Reda"
    author_additional = ["Janne Jontkanen", "Eric Tabellion", "Deqing Sun", "Caroline Pantofaru", "Brian Curless"]
    author_url = "https://arxiv.org/abs/2202.04901"
    author_journal = "ECCV"
    author_journal_year = 2022
    author_journal_title = "FiLM: Frame Interpolation for Large Motion"
    author_affiliations = ["Google Research", "University of Washington"]

    """License metadata"""
    license = LICENSE_APACHE

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Required packages.
        """
        return {
            "pil": PILLOW_VERSION_SPEC,
            "torch": TORCH_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
        }

    """Internal Task Attributes"""

    @property
    def module(self) -> torch.nn.Module:
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
        import bisect
        import torch
        import numpy as np
        start, crop_start = pad_image_to_nearest(start, 64, return_crop=True) # type: ignore[misc, assignment]
        end, crop_end = pad_image_to_nearest(end, 64, return_crop=True) # type: ignore[misc, assignment]

        assert crop_start == crop_end, "Images must be the same size"
        l, t, r, b = crop_start # type: ignore[misc]

        indexes = [0, num_frames + 1]
        remains = list(range(1, num_frames + 1))
        splits = torch.linspace(0, 1, num_frames + 2)
        results = [start, end]

        for i in range(len(remains)):
            starts = splits[indexes[:-1]]
            ends = splits[indexes[1:]]
            distances = (
                (splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5
            ).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape) # type: ignore[arg-type]
            end_i = start_i + 1

            x_0 = results[start_i]
            x_1 = results[end_i]

            d_t = x_0.new_full(
                (1, 1),
                (splits[remains[step]] - splits[indexes[start_i]]) # type: ignore[arg-type]
            ) / (splits[indexes[end_i]] - splits[indexes[start_i]])

            pred = self.module(x_0, x_1, d_t)

            insert_position = bisect.bisect_left(indexes, remains[step])
            indexes.insert(insert_position, remains[step])
            results.insert(insert_position, pred.clamp(0,1))

            del remains[step]

        return torch.cat([
            result.cpu()[:, t:b, l:r] # type: ignore[misc]
            for result in results
        ], dim=0)

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
        Interpolates two images with FiLM (Frame Interpolation for Large Motion).
        :param start: The first image to interpolate between.
        :param end: The second image to interpolate between.
        :param num_frames: The number of frames to interpolate between the two images.
        :param include_ends: Whether to include the start and end frames in the output.
        :param output_format: The format of the output image.
        :param output_upload: Whether to upload the output image to the configured storage backend and return the URL, or return the image data directly.
        :return: The interpolated frames.
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
