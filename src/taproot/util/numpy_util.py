from __future__ import annotations

from typing import Any, Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "hwc3",
    "safe_resize",
    "nms_mask",
    "safe_step",
    "NumpyDataBuffer",
]

def hwc3(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Ensures an image is in HWC format with 3 channels.
    """
    import numpy as np
    assert image.dtype == np.uint8
    if image.ndim == 2:
        image = image[:, :, None]
    assert image.ndim == 3
    h, w, c = image.shape
    assert c in [1, 3, 4]
    if c == 3:
        return image
    elif c == 1:
        return np.concatenate([image] * 3, axis=2)
    else:
        color = image[:, :, 0:3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        combined = color * alpha + 255 * (1 - alpha)
        combined = combined.clip(0, 255).astype(np.uint8)
        return combined # type: ignore[no-any-return]

def safe_resize(image: np.ndarray[Any, Any], resolution: int, nearest: int=64) -> np.ndarray[Any, Any]:
    """
    Resizes an image to the specified resolution, padding if necessary.
    """
    import cv2 # type: ignore[import-not-found]
    import numpy as np
    h, w, c = image.shape
    k = float(resolution) / min(h, w)
    h = float(h) * k # type: ignore[assignment]
    w = float(w) * k # type: ignore[assignment]
    h = int(np.round(h / float(nearest)) * nearest)
    w = int(np.round(w / float(nearest)) * nearest)
    image = cv2.resize(
        image,
        (w, h),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    )
    return image

def nms_mask(
    mask_image: np.ndarray[Any, Any],
    threshold: int=127,
    sigma: float=3.0
) -> np.ndarray[Any, Any]:
    """
    Performs non-maximum suppression on a mask image.
    """
    import cv2
    import numpy as np
    mask_image = cv2.GaussianBlur(mask_image, (0, 0), sigma)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    zero = np.zeros_like(mask_image)
    for f in [f1, f2, f3, f4]:
        np.putmask(
            zero,
            cv2.dilate(mask_image, kernel=f) == mask_image,
            mask_image
        )
    mask_image = np.zeros_like(zero, dtype=np.uint8)
    mask_image[zero > threshold] = 255
    return mask_image

def safe_step(image: np.ndarray[Any, Any], step: int=2) -> np.ndarray[Any, Any]:
    """
    Steps an image by the specified amount.
    """
    import numpy as np
    stepped = image.astype(np.float32) * float(step + 1)
    stepped = stepped.astype(np.int32).astype(np.float32) / float(step)
    return stepped

class NumpyDataBuffer:
    """
    A fast, circular FIFO buffer in numpy with minimal memory interactions by using an array of index pointers
    """
    def __init__(
        self,
        n_windows: int,
        samples_per_window: int,
        dtype: Optional[Type[np.number[Any]]] = None,
        start_value: int = 0,
        data_dimensions: int = 1
    ) -> None:
        import numpy as np
        if dtype is None:
            dtype = np.float32
        self.n_windows = n_windows
        self.data_dimensions = data_dimensions
        self.samples_per_window = samples_per_window
        self.data = start_value * np.ones((self.n_windows, self.samples_per_window), dtype = dtype)

        if self.data_dimensions == 1:
            self.total_samples = self.n_windows * self.samples_per_window
        else:
            self.total_samples = self.n_windows

        self.elements_in_buffer = 0
        self.overwrite_index = 0

        self.indices = np.arange(self.n_windows, dtype=np.int32)
        self.last_window_id = np.max(self.indices)
        self.index_order = np.argsort(self.indices)

    def append_data(self, data_window: np.ndarray[Any, Any]) -> None:
        """
        Appends data to the buffer.
        """
        import numpy as np
        self.data[self.overwrite_index, :] = data_window

        self.last_window_id += 1
        self.indices[self.overwrite_index] = self.last_window_id
        self.index_order = np.argsort(self.indices)

        self.overwrite_index += 1
        self.overwrite_index = self.overwrite_index % self.n_windows

        self.elements_in_buffer += 1
        self.elements_in_buffer = min(self.n_windows, self.elements_in_buffer)

    def get_most_recent(self, window_size: int) -> np.ndarray[Any, Any]:
        """
        Gets the most recent data from the buffer.
        """
        import numpy as np
        ordered_dataframe = self.data[self.index_order]
        if self.data_dimensions == 1:
            ordered_dataframe = np.hstack(ordered_dataframe) # type: ignore[call-overload,unused-ignore]
        return ordered_dataframe[self.total_samples - window_size:] # type: ignore[no-any-return,unused-ignore]

    def get_buffer_data(self) -> np.ndarray[Any, Any]:
        """
        Gets the data from the buffer.
        """
        return self.data[:self.elements_in_buffer]
