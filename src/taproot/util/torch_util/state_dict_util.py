from __future__ import annotations

from typing import Dict, Union, Optional, Iterable, Tuple, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, device as Device, dtype as DType
    import torch.nn as nn

__all__ = [
    "load_ckpt_state_dict",
    "load_safetensor_state_dict",
    "load_state_dict",
    "iterate_state_dict",
    "inject_state_dict",
    "save_state_dict",
]

def set_state_dict_dtype(state_dict: Dict[str, Any], dtype: DType) -> None:
    """
    Sets state dict data type in place.
    """
    import torch
    for key, value in state_dict.items():
        if isinstance(value, dict):
            set_state_dict_dtype(value, dtype)
        elif isinstance(value, torch.Tensor):
            state_dict[key] = value.to(dtype=dtype)

def load_ckpt_state_dict(
    path: str,
    device: Union[str, Device]="cpu",
    dtype: Optional[DType]=None,
) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
    """
    Loads a state dictionary from a .ckpt (old-style) file
    """
    import torch
    state_dict = torch.load(path, map_location=device)
    while "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if dtype is not None:
        set_state_dict_dtype(state_dict, dtype)
    return state_dict # type: ignore[no-any-return]

def load_safetensor_state_dict(
    path: str,
    device: Union[str, Device]="cpu",
    dtype: Optional[DType]=None,
) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
    """
    Loads a state dictionary from a .safetensor(s) (new-style) file
    """
    from safetensors import safe_open

    checkpoint = {}
    with safe_open(path, framework="pt", device=str(device)) as f: # type: ignore
        for key in f.keys():
            checkpoint[key] = f.get_tensor(key)
            if dtype is not None:
                checkpoint[key] = checkpoint[key].to(dtype=dtype)
    return checkpoint

def save_state_dict(
    path: str,
    state_dict: Dict[str, Tensor],
    metadata: Optional[Dict[str, str]]=None
) -> None:
    """
    Saves a state dictionary to file.
    """
    if "safetensor" in path:
        import safetensors
        import safetensors.torch
        safetensors.torch.save_file(
            state_dict,
            path,
            metadata=metadata
        )
    else:
        import torch
        torch.save(state_dict, path)

def load_state_dict(
    path: str,
    device: Optional[Union[str, Device]]="cpu",
    dtype: Optional[DType]=None,
) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
    """
    Loads a state dictionary from file.
    Tries to correct issues with incorrrect formats.
    """
    load_order = [load_safetensor_state_dict, load_ckpt_state_dict]
    if "safetensor" not in path:
        load_order = [load_ckpt_state_dict, load_safetensor_state_dict]

    if device is None:
        device = "cpu"

    first_error: Optional[Exception] = None
    for i, loader in enumerate(load_order):
        try:
            return loader(path, device=device, dtype=dtype)
        except Exception as ex:
            if first_error is None:
                first_error = ex

    if first_error is not None:
        raise IOError(f"Received exception reading checkpoint {path}, please ensure file integrity.\n{type(first_error).__name__}: {first_error}")
    raise IOError(f"No data read from path {path}")

def iterate_state_dict(
    path: str,
    device: Union[str, Device]="cpu",
    dtype: Optional[DType]=None,
) -> Iterable[Tuple[str, Tensor]]:
    """
    Loads a state dict one tensor at a time.
    """
    if "safetensor" not in path:
        sd = load_state_dict(path, dtype=dtype)
        for key in sd:
            yield (key, sd[key]) # type: ignore[misc]
    else:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device=str(device)) as f: # type: ignore
            for key in f.keys():
                t = f.get_tensor(key)
                if dtype is not None:
                    t = t.to(dtype=dtype)
                yield (key, t)

def inject_state_dict(
    path: str,
    target: nn.Module,
    device: Union[str, Device]="cpu",
    dtype: Optional[DType]=None,
    strict: bool=True
) -> List[str]:
    """
    Iterates over a state dictionary and injects it into a module using accelerate.
    """
    from accelerate.utils import set_module_tensor_to_device # type: ignore[import-not-found,import-untyped,unused-ignore]
    unexpected = []
    for key, value in iterate_state_dict(path, device=device, dtype=dtype):
        try:
            set_module_tensor_to_device(target, key, str(device), value)
        except (AttributeError, ValueError):
            if strict:
                raise
            else:
                unexpected.append(key)
    return unexpected
