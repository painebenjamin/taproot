from __future__ import annotations

from typing import Optional, Tuple, Union, Tuple, Any, TYPE_CHECKING

from ...constants import *
from ..introspection_util import realize_kwargs
from ..terminal_util import maybe_use_tqdm
from ..misc_util import sliding_windows
from ..torch_util import MaskWeightBuilder
from ..prompt_util import EncodedPrompts
from ..log_util import logger

if TYPE_CHECKING:
    import torch

__all__ = ["enable_2d_multidiffusion", "disable_2d_multidiffusion"]

ORIGINAL_FORWARD_ATTRIBUTE_NAME = "_single_diffusion_forward"

def enable_2d_multidiffusion(
    model: torch.nn.Module,
    spatial_prompts: Optional[EncodedPrompts]=None,
    tile_size: Optional[Union[int, Tuple[int, int]]]=None,
    tile_stride: Optional[Union[int, Tuple[int, int]]]=None,
    use_tqdm: bool=False,
    mask_type: MULTIDIFFUSION_MASK_TYPE_LITERAL=DEFAULT_MULTIDIFFUSION_MASK_TYPE, # type: ignore[assignment]
) -> None:
    """
    Patch a 2D diffusion model to support multi-diffusion.

    Should work on any model that takes image tensors as input and returns image tensors as output,
    either in pixel space or in latent space.

    Presently does not work with packed/patched tensor input (i.e., does not work with FLUX.)
    """
    import torch

    # Standardize size and stride
    if isinstance(tile_size, int):
        tile_width = tile_height = tile_size
    elif isinstance(tile_size, tuple):
        tile_width, tile_height = tile_size
    else:
        if hasattr(model.config, "sample_size"):
            tile_width = tile_height = model.config.sample_size
        else:
            tile_width = tile_height = 128

    tile_size = (tile_width, tile_height)

    if isinstance(tile_stride, int):
        tile_stride_width = tile_stride_height = tile_stride
    elif isinstance(tile_stride, tuple):
        tile_stride_width, tile_stride_height = tile_stride
    else:
        tile_stride_width = tile_width // 2
        tile_stride_height = tile_height // 2

    tile_stride = (tile_stride_width, tile_stride_height)

    # Get device and dtype
    if hasattr(model, "device"):
        device = model.device
    else:
        device = next(model.parameters()).device

    if hasattr(model, "dtype"):
        dtype = model.dtype
    else:
        dtype = next(model.parameters()).dtype

    logger.debug(f"Enabling 2D multi-diffusion with tile size {tile_size}, stride {tile_stride}, and mask type {mask_type} on {type(model).__name__}")

    # Initialize mask builder
    mask_builder = MaskWeightBuilder(device=device, dtype=dtype)

    # Store original forward method
    if not hasattr(model, ORIGINAL_FORWARD_ATTRIBUTE_NAME):
        setattr(model, ORIGINAL_FORWARD_ATTRIBUTE_NAME, model.forward)

    original_forward = getattr(model, ORIGINAL_FORWARD_ATTRIBUTE_NAME)

    # Define new forward method
    def forward(*args: Any, **kwargs: Any) -> Any:
        """
        Wrap the model's forward method to support multi-diffusion.
        """
        # Standardize as kwargs dictionary
        kwargs = realize_kwargs(original_forward, args, kwargs)

        # Identify image tensors that we can window over
        image_kwargs = [
            key for key, value in kwargs.items()
            if isinstance(value, torch.Tensor) and value.ndim == 4
        ]

        # If there are image tensors, tile them and apply the model to each tile
        if image_kwargs:
            original_tensors = [kwargs[key] for key in image_kwargs]
            b, c, h, w = kwargs[image_kwargs[0]].shape
            windows = sliding_windows(
                height=h,
                width=w,
                tile_size=tile_size,
                tile_stride=tile_stride
            )
            if len(windows) <= 1:
                # No tiling
                return original_forward(**kwargs)

            result: Any = None
            result_count: Any = None
            result_tuple = False

            for (top, bottom, left, right) in maybe_use_tqdm(windows, use_tqdm, desc="Diffusing tiles"):
                for image_kwarg, original_tensor in zip(image_kwargs, original_tensors):
                    kwargs[image_kwarg] = original_tensor[:, :, top:bottom, left:right]

                # Inject postional prompts
                if spatial_prompts is not None:
                    embeddings, pooled_embeddings = spatial_prompts.get_embeddings(
                        position=(left, top, right, bottom),
                        device=device,
                        dtype=dtype,
                    )

                    kwargs["encoder_hidden_states"] = embeddings
                    if "pooled_projections" in kwargs:
                        kwargs["pooled_projections"] = pooled_embeddings

                result_tile = original_forward(**kwargs)
                result_mask = mask_builder(
                    mask_type,
                    batch=b,
                    dim=c,
                    width=right - left,
                    height=bottom - top,
                    unfeather_left=left == 0,
                    unfeather_top=top == 0,
                    unfeather_right=right == w,
                    unfeather_bottom=bottom == h,
                )

                # Initialize result container if not already initialized
                if result is None:
                    if isinstance(result_tile, torch.Tensor):
                        # single tensor output
                        result = torch.zeros((b, c, h, w), dtype=result_tile.dtype, device=result_tile.device)
                        result_count = result.clone()
                    elif isinstance(result_tile, (tuple, list)):
                        # Multiple value output
                        result = [
                            torch.zeros((b, c, h, w), dtype=value.dtype, device=value.device)
                            if isinstance(value, torch.Tensor)
                            else value
                            for value in result_tile
                        ]
                        result_count = [
                            value.clone()
                            if isinstance(value, torch.Tensor)
                            else None
                            for value in result
                        ]
                        result_tuple = isinstance(result_tile, tuple)
                    elif isinstance(result_tile, dict):
                        result = {
                            key: (torch.zeros((b, c, h, w), dtype=value.dtype, device=value.device) if isinstance(value, torch.Tensor) else value)
                            for key, value in result_tile.items()
                        }
                        result_count = {
                            key: (value.clone() if isinstance(value, torch.Tensor) else None)
                            for key, value in result.items()
                        }
                    else:
                        raise ValueError(f"Unsupported output type {type(result_tile)}")

                # Update result container
                if isinstance(result, torch.Tensor):
                    result[:, :, top:bottom, left:right] = result[:, :, top:bottom, left:right] + result_tile * result_mask
                    result_count[:, :, top:bottom, left:right] = result_count[:, :, top:bottom, left:right] + result_mask
                elif isinstance(result, list):
                    for i, result_part in enumerate(result_tile):
                        if isinstance(result_part, torch.Tensor):
                            result[i][:, :, top:bottom, left:right] = result[i][:, :, top:bottom, left:right] + result_part * result_mask
                            result_count[i][:, :, top:bottom, left:right] = result_count[i][:, :, top:bottom, left:right] + result_mask
                elif isinstance(result, dict):
                    for key, result_part in result_tile.items():
                        if isinstance(result_part, torch.Tensor):
                            result[key][:, :, top:bottom, left:right] = result[key][:, :, top:bottom, left:right] + result_part * result_mask
                            result_count[key][:, :, top:bottom, left:right] = result_count[key][:, :, top:bottom, left:right] + result_mask

            # Normalize result and return
            if isinstance(result, torch.Tensor):
                result = result / result_count
            elif isinstance(result, list):
                result = [
                    value / result_count[i]
                    if isinstance(value, torch.Tensor)
                    else value
                    for i, value in enumerate(result)
                ]
                if result_tuple:
                    result = tuple(result)
            elif isinstance(result, dict):
                result = {
                    key: (value / result_count[key] if isinstance(value, torch.Tensor) else value)
                    for key, value in result.items()
                }
            return result
        else:
            return original_forward(**kwargs)

    # Set new forward method
    setattr(model, "forward", forward)

def disable_2d_multidiffusion(model: torch.nn.Module) -> None:
    """
    Disable multi-diffusion support for a 2D diffusion pipeline.

    If multi-diffusion support is not enabled, this function does nothing.
    """
    if hasattr(model, ORIGINAL_FORWARD_ATTRIBUTE_NAME):
        setattr(model, "forward", getattr(model, ORIGINAL_FORWARD_ATTRIBUTE_NAME))
