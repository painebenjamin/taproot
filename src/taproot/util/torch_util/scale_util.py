from __future__ import annotations

from typing import Optional, Tuple, Union, TYPE_CHECKING
from typing_extensions import Literal

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["scale_tensor"]

def scale_tensor(
    arg: Tensor,
    round_to_nearest: Optional[int] = None,
    scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    upscale_mode: Literal["nearest-exact", "linear", "bilinear", "trilinear", "bicubic"] = "bicubic",
    upscale_antialias: bool = True,
    downscale_mode: Literal["nearest-exact", "linear", "bilinear", "trilinear", "bicubic", "area", "pool-max", "pool-avg"] = "area",
    downscale_antialias: bool = True
) -> Tensor:
    """
    Scale a spatial tensor (3D, 4D, 5D) using the specified scale factor or size.
    Either (b, c, l), (b, c, h, w), or (b, c, d, h, w) shaped tensors are supported.

    >>> import torch
    >>> scale_tensor(torch.randn(1, 3, 256, 256), scale_factor=0.5).shape
    torch.Size([1, 3, 128, 128])
    >>> scale_tensor(torch.randn(1, 3, 256, 256), scale_factor=(0.5, 0.75)).shape
    torch.Size([1, 3, 128, 192])
    >>> scale_tensor(torch.randn(1, 3, 256, 256), size=128).shape
    torch.Size([1, 3, 128, 128])
    >>> scale_tensor(torch.randn(1, 3, 256, 256), size=(196, 128)).shape
    torch.Size([1, 3, 196, 128])
    >>> scale_tensor(torch.randn(1, 3, 134, 190), scale_factor=1.0, round_to_nearest=64).shape
    torch.Size([1, 3, 128, 192])
    >>> scale_tensor(torch.randn(1, 3, 256, 256), size=(400, 400), round_to_nearest=64).shape
    torch.Size([1, 3, 384, 384])
    """
    if round_to_nearest is not None and scale_factor is None and size is None:
        scale_factor = 1.0 # Just round
    assert scale_factor is not None or size is not None, "Either scale_factor or size must be specified"
    assert scale_factor is None and size is not None or scale_factor is not None and size is None, "Only one of scale_factor or size can be specified"
    assert upscale_mode in ["nearest-exact", "linear", "bilinear", "trilinear", "bicubic"], "Invalid upscale mode"
    assert downscale_mode in ["nearest-exact", "linear", "bilinear", "trilinear", "bicubic", "area", "pool-max", "pool-avg"], "Invalid downscale mode"
    import torch.nn.functional as F

    if round_to_nearest is not None:
        if size is not None:
            if isinstance(size, tuple):
                h, w = size
            else:
                h = w = size
            h = round(h / round_to_nearest) * round_to_nearest
            w = round(w / round_to_nearest) * round_to_nearest
            size = (h, w)
        else:
            # Calculate the size beforehand
            h, w = arg.shape[-2:]
            if isinstance(scale_factor, tuple):
                h_scale, w_scale = scale_factor
            elif isinstance(scale_factor, float):
                h_scale = w_scale = scale_factor
            else:
                h_scale = w_scale = 1.0
            h = round(h * h_scale / round_to_nearest) * round_to_nearest
            w = round(w * w_scale / round_to_nearest) * round_to_nearest
            size = (h, w)
            scale_factor = None

    if scale_factor is not None:
        if isinstance(scale_factor, float):
            if scale_factor == 1.0:
                return arg
            is_upscale = scale_factor > 1.0
        elif isinstance(scale_factor, tuple):
            scale_h, scale_w = scale_factor
            if scale_h == 1.0 and scale_w == 1.0:
                return arg
            is_upscale = scale_h * scale_w > 1.0
        else:
            raise ValueError("Scale factor must be a singular float value or a tuple of two float values")
    else:
        h, w = arg.shape[-2:]
        if isinstance(size, int):
            if size == h and size == w:
                return arg
            is_upscale = size / h * size / w > 1.0
        elif isinstance(size, tuple):
            size_h, size_w = size
            if size_h == h and size_w == w:
                return arg
            is_upscale = size_h / h * size_w / w > 1.0
        else:
            raise ValueError("Size must be an integer value or a tuple of two integer values")

    mode = upscale_mode if is_upscale else downscale_mode
    antialias = upscale_antialias if is_upscale else downscale_antialias

    if mode in ["pool-max", "pool-avg"]:
        assert size is None, "Size cannot be specified for pool downsampling"
        assert isinstance(scale_factor, float), "Scale factor must be a singular float value for pooled downsampling"
        kernel_size = int(1.0 / scale_factor)
        if mode == "pool-max":
            return F.max_pool2d(arg, kernel_size=kernel_size)
        return F.avg_pool2d(arg, kernel_size=kernel_size)

    if mode in ["trilinear", "bilinear", "linear"]:
        ndim = arg.ndim - 2
        if ndim == 1:
            mode = "linear"
        elif ndim == 2:
            mode = "bilinear"
        else:
            mode = "trilinear"
    else:
        antialias = False

    no_batch_dim = arg.ndim == 3

    result = F.interpolate(
        arg.unsqueeze(0) if no_batch_dim else arg,
        scale_factor=scale_factor,
        size=size,
        mode=mode,
        antialias=antialias
    )
    if no_batch_dim:
        return result[0] # type: ignore[no-any-return]
    return result # type: ignore[no-any-return]
