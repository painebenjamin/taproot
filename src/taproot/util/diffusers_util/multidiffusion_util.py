from __future__ import annotations

from typing import Optional, Tuple, Union, Tuple, Any, Sequence, TYPE_CHECKING

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
    is_packed: bool=False,
    input_keys: Optional[Sequence[str]]=None,
    mask_type: MULTIDIFFUSION_MASK_TYPE_LITERAL=DEFAULT_MULTIDIFFUSION_MASK_TYPE, # type: ignore[assignment]
) -> None:
    """
    Patch a 2D diffusion model to support multi-diffusion.

    Should work on any model that takes image tensors as input and returns image tensors as output,
    either in pixel space or in latent space.
    """
    import torch

    if is_packed:
        if isinstance(tile_size, (tuple, list)):
            tile_size = tile_size[0]
        elif tile_size is None:
            if hasattr(model.config, "joint_attention_dim"):
                tile_size = model.config.joint_attention_dim # type: ignore[assignment]
            else:
                tile_size = 4096

        if isinstance(tile_stride, (tuple, list)):
            tile_stride = tile_stride[0]
        elif tile_stride is None:
            tile_stride = tile_size // 2 # type: ignore[operator]
    else:
        # Standardize size and stride
        if isinstance(tile_size, int):
            tile_width = tile_height = tile_size
        elif isinstance(tile_size, tuple):
            tile_width, tile_height = tile_size
        else:
            if hasattr(model.config, "sample_size"):
                tile_width = tile_height = model.config.sample_size # type: ignore[assignment]
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
        device = next(model.parameters()).device # type: ignore[assignment]

    if hasattr(model, "dtype"):
        dtype = model.dtype
    else:
        dtype = next(model.parameters()).dtype # type: ignore[assignment]

    logger.debug(f"Enabling 2D multi-diffusion with tile size {tile_size}, stride {tile_stride}, and mask type {mask_type} on {type(model).__name__}")

    # Initialize mask builder
    mask_builder = MaskWeightBuilder(device=device, dtype=dtype) # type: ignore[arg-type]

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
            if isinstance(value, torch.Tensor) and value.ndim == (3 if is_packed else 4)
            and (input_keys is None or key in input_keys)
        ]

        # If there are image tensors, tile them and apply the model to each tile
        if image_kwargs:
            original_tensors = [kwargs[key] for key in image_kwargs]
            image_ids = kwargs.get("img_ids", None)
            if is_packed:
                b, w, c = kwargs[image_kwargs[0]].shape
                h = None
            else:
                b, c, h, w = kwargs[image_kwargs[0]].shape

            windows = sliding_windows(
                height=h,
                width=w,
                tile_size=tile_size, # type: ignore[arg-type]
                tile_stride=tile_stride
            )

            if len(windows) <= 1 and spatial_prompts is None:
                # No tiling or masking needed
                return original_forward(**kwargs)

            result: Any = None
            result_count: Any = None
            result_tuple = False

            for i, window in enumerate(maybe_use_tqdm(windows, use_tqdm, desc="Diffusing tiles")):
                if is_packed:
                    (left, right) = window
                    top = bottom = 0
                    for image_kwarg, original_tensor in zip(image_kwargs, original_tensors):
                        kwargs[image_kwarg] = original_tensor[:, left:right, :]
                    if image_ids is not None: # Pixel positions of packed images
                        kwargs["img_ids"] = image_ids[left:right]
                else:
                    (top, bottom, left, right) = window

                    for image_kwarg, original_tensor in zip(image_kwargs, original_tensors):
                        kwargs[image_kwarg] = original_tensor[:, :, top:bottom, left:right]

                # Inject postional prompts
                embeddings: Optional[torch.Tensor] = None
                pooled_embeddings: Optional[torch.Tensor] = None
                mask: Optional[torch.Tensor] = None
                if spatial_prompts is not None:
                    embeddings, pooled_embeddings, mask = spatial_prompts.get_embeddings(
                        position=(left, top, right, bottom),
                        device=device, # type: ignore[arg-type]
                        dtype=dtype, # type: ignore[arg-type]
                    )

                if mask is not None:
                    # Prompts have spatial masks, we will use them to mask each tile
                    num_masks = mask.shape[0]
                else:
                    num_masks = 1

                for j in range(num_masks):
                    if embeddings is not None:
                        if mask is not None:
                            kwargs["encoder_hidden_states"] = embeddings[j]
                        else:
                            kwargs["encoder_hidden_states"] = embeddings
                    if "pooled_projections" in kwargs and pooled_embeddings is not None:
                        if mask is not None:
                            kwargs["pooled_projections"] = pooled_embeddings[j]
                        else:
                            kwargs["pooled_projections"] = pooled_embeddings

                    result_tile = original_forward(**kwargs)
                    result_mask = mask_builder(
                        mask_type,
                        batch=b,
                        dim=c,
                        width=right - left,
                        height=None if is_packed else bottom - top,
                        unfeather_left=left == 0,
                        unfeather_top=not is_packed and top == 0,
                        unfeather_right=right == w,
                        unfeather_bottom=not is_packed and bottom == h,
                    )

                    # Initialize result container if not already initialized
                    if result is None:
                        if isinstance(result_tile, torch.Tensor):
                            # single tensor output
                            if is_packed:
                                result = torch.zeros((b, w, c), dtype=result_tile.dtype, device=result_tile.device)
                            else:
                                result = torch.zeros((b, c, h, w), dtype=result_tile.dtype, device=result_tile.device) # type: ignore[arg-type]
                            result_count = result.clone()
                        elif isinstance(result_tile, (tuple, list)):
                            # Multiple value output
                            if is_packed:
                                result = [
                                    torch.zeros((b, w, c), dtype=value.dtype, device=value.device)
                                    if isinstance(value, torch.Tensor)
                                    else value
                                    for value in result_tile
                                ]
                            else:
                                result = [
                                    torch.zeros((b, c, h, w), dtype=value.dtype, device=value.device) # type: ignore[arg-type]
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
                            if is_packed:
                                result = {
                                    key: (
                                        torch.zeros((b, w, c), dtype=value.dtype, device=value.device)
                                        if isinstance(value, torch.Tensor)
                                        else value
                                    )
                                    for key, value in result_tile.items()
                                }
                            else:
                                result = {
                                    key: (
                                        torch.zeros((b, c, h, w), dtype=value.dtype, device=value.device) # type: ignore[arg-type]
                                        if isinstance(value, torch.Tensor)
                                        else value
                                    )
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
                        if is_packed:
                            result[:, left:right, :] = result[:, left:right, :] + result_tile * result_mask
                            result_count[:, left:right, :] = result_count[:, left:right, :] + result_mask
                        else:
                            result[:, :, top:bottom, left:right] = result[:, :, top:bottom, left:right] + result_tile * result_mask
                            result_count[:, :, top:bottom, left:right] = result_count[:, :, top:bottom, left:right] + result_mask
                    elif isinstance(result, list):
                        for k, result_part in enumerate(result_tile):
                            if isinstance(result_part, torch.Tensor):
                                if is_packed:
                                    result[k][:, left:right, :] = result[k][:, left:right, :] + result_part * result_mask
                                    result_count[k][:, left:right, :] = result_count[k][:, left:right, :] + result_mask
                                else:
                                    result[k][:, :, top:bottom, left:right] = result[k][:, :, top:bottom, left:right] + result_part * result_mask
                                    result_count[k][:, :, top:bottom, left:right] = result_count[k][:, :, top:bottom, left:right] + result_mask
                    elif isinstance(result, dict):
                        for key, result_part in result_tile.items():
                            if isinstance(result_part, torch.Tensor):
                                if is_packed:
                                    result[key][:, left:right, :] = result[key][:, left:right, :] + result_part * result_mask
                                    result_count[key][:, left:right, :] = result_count[key][:, left:right, :] + result_mask
                                else:
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
