from __future__ import annotations

from taproot.util import to_bchw_tensor, maybe_use_tqdm
from taproot.constants import *
from taproot.tasks.base import Task

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    from taproot.hinting import ImageType, ImageResultType

__all__ = ["VideoFrameInterpolationTaskBase"]

class VideoFrameInterpolationTaskBase(Task):
    """
    Base class for video frame interpolation tasks.

    These are generally wrappers for their image interpolation counterparts.
    """

    """Internal Task Attributes"""

    @property
    def interpolator(self) -> Task:
        """
        Add mapped modules as properties for convenience and type hinting.
        """
        try:
            return self.tasks.interpolate
        except Exception as e:
            raise AttributeError("Interpolator not available - did you configure `component_tasks[interpolate]`?") from e

    """Internal Methods"""

    def interpolate(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        multiplier: int
    ) -> torch.Tensor:
        """
        Invokes the interpolator task with the given tensors.

        :param left: The left image tensor.
        :param right: The right image tensor.
        :param multiplier: The number of frames to interpolate between the two images.
        :return: The interpolated tensor.
        """
        import torch
        interpolated = self.interpolator(
            start=left,
            end=right,
            num_frames=multiplier,
            include_ends=False,
            output_format="float",
            output_upload=False
        )
        assert isinstance(interpolated, torch.Tensor)
        return interpolated.detach().cpu().float()

    """Overrides"""

    def __call__( # type: ignore[override]
        self,
        *,
        video: ImageType,
        loop: bool=False,
        multiplier: int=2,
        frame_rate: int=DEFAULT_FRAME_RATE,
        output_format: VIDEO_OUTPUT_FORMAT_LITERAL="mp4",
        output_upload: bool=False,
    ) -> ImageResultType:
        """
        Interpolates frames in a video.

        :param video: The input video.
        :param loop: Whether to loop the video.
        :param multiplier: The number of frames to interpolate between each pair of frames.
        :param frame_rate: The frame rate of the output video.
        :param output_format: The output video format.
        :param output_upload: Whether to upload the output video.
        :return: The interpolated video.
        """
        import torch
        with torch.inference_mode():
            # Use utility methods to standardize the input
            images = to_bchw_tensor(video, num_channels=3)
            num_frames, _, height, width = images.shape
            num_output_frames = num_frames * multiplier

            if not loop:
                num_output_frames -= multiplier

            results = torch.zeros(
                (num_output_frames, 3, height, width),
                device=torch.device("cpu"),
                dtype=torch.float32
            )

            for i in maybe_use_tqdm(range(num_frames), desc="Interpolating frames", total=num_frames):
                left = images[i]

                if i == num_frames - 1:
                    if not loop:
                        break
                    right = images[0]
                else:
                    right = images[i + 1]

                start_i = i * multiplier

                results[start_i] = left
                results[start_i:start_i + multiplier] = self.interpolate(
                    left=left,
                    right=right,
                    multiplier=multiplier
                )

            if not loop:
                # Add final frame
                results[-1] = images[-1]

        # This utility method will get the requested format
        return self.get_output_from_video_result(
            results.unsqueeze(0),
            multi_video=False,
            frame_rate=frame_rate,
            output_format=output_format,
            output_upload=output_upload,
        )
