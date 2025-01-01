from __future__ import annotations

from typing import Tuple, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import device as Device
    from torch.nn import Module

__all__ = [
    "cuda_available",
    "tensorrt_available",
    "mps_available",
    "directml_available",
    "get_optimal_device",
    "get_ram_info",
    "get_vram_info",
    "empty_cache",
    "debug_tensors",
    "set_max_split_size_mb",
    "llama_cpp_available",
]

def tensorrt_available() -> bool:
    """
    Returns true if TensorRT is available.
    """
    try:
        import tensorrt # type: ignore[import-not-found,unused-ignore]
        tensorrt  # silence importchecker
        import onnx # type: ignore[import-not-found,unused-ignore]
        onnx # silence importchecker
        import onnx_graphsurgeon # type: ignore[import-not-found]
        onnx_graphsurgeon # silence importchecker
        return True
    except:
        return False

def torch_imported() -> bool:
    """
    Returns true if torch is imported by a brittle sys.modules check.
    """
    import sys
    return "torch" in sys.modules

def cuda_available() -> bool:
    """
    Returns true if CUDA is available.
    """
    import torch
    return torch.cuda.is_available() and torch.backends.cuda.is_built() # type: ignore[no-untyped-call]

def mps_available() -> bool:
    """
    Returns true if MPS is available.
    """
    import torch
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()

def directml_available() -> bool:
    """
    Returns true if directml is available.
    """
    try:
        import torch_directml # type: ignore[import-not-found]
        return True
    except:
        return False

def get_optimal_device(device_index: Optional[int] = None) -> Device:
    """
    Gets the optimal device based on availability.
    """
    import torch
    if cuda_available():
        return torch.device("cuda", 0 if device_index is None else device_index)
    elif directml_available():
        import torch_directml
        return torch_directml.device() # type: ignore[no-any-return]
    elif mps_available():
        return torch.device("mps", 0 if device_index is None else device_index)
    return torch.device("cpu")

def get_ram_info() -> Tuple[int, int]:
    """
    Returns RAM amount in bytes as [free, total]
    """
    import psutil
    mem = psutil.virtual_memory()
    return (mem.free, mem.total)

def get_vram_info() -> Tuple[int, int]:
    """
    Returns VRAM amount in bytes as [free, total]
    If no GPU is found, returns RAM info.
    """
    if not cuda_available():
        return get_ram_info()
    import torch
    return torch.cuda.mem_get_info()

def empty_cache(synchronize: bool=True) -> None:
    """
    Empties caches to clear memory.
    """
    import gc
    gc.collect()
    if torch_imported():
        if cuda_available():
            import torch
            import torch.cuda
            torch.cuda.empty_cache()
            if synchronize:
                torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        elif mps_available():
            import torch
            import torch.mps
            torch.mps.empty_cache()
            if synchronize:
                torch.mps.synchronize()

def debug_tensors(*args: Any, **kwargs: Any) -> None:
    """
    Logs tensors
    """
    import torch
    from ..log_util import logger
    include_bounds = kwargs.pop("include_bounds", False)
    arg_dict = dict([
        (f"arg_{i}", arg)
        for i, arg in enumerate(args)
    ])
    for tensor_dict in [arg_dict, kwargs]:
        for key, value in tensor_dict.items():
            if isinstance(value, list) or isinstance(value, tuple):
                for i, v in enumerate(value):
                    debug_tensors(include_bounds=include_bounds, **{f"{key}_{i}": v})
            elif isinstance(value, dict):
                for k, v in value.items():
                    debug_tensors(include_bounds=include_bounds, **{f"{key}_{k}": v})
            elif isinstance(value, torch.Tensor):
                if include_bounds:
                    t_min, t_max = value.aminmax()
                    logger.debug(f"{key} = {value.shape} ({value.dtype}) on {value.device}, min={t_min}, max={t_max}")
                else:
                    logger.debug(f"{key} = {value.shape} ({value.dtype}) on {value.device}")

def set_max_split_size_mb(
    module: Module,
    max_split_size_mb: int,
    enable_grad: bool=False
) -> None:
    """
    Sets the maximum split size in the model.
    """
    import torch
    for param in module.parameters():
        param.requires_grad = False

    try:
        module(torch.randn(1,1))
    except:
        pass

    allocator = torch.cuda.memory._get_memory_allocator() # type: ignore[attr-defined]
    allocator.set_max_split_size(max_split_size_mb * 1024 * 1024)

    if enable_grad:
        for param in module.parameters():
            param.requires_grad = True

def llama_cpp_available() -> bool:
    """
    Returns true if the llama_cpp library is available.
    """
    if not hasattr(llama_cpp_available, "_available"):
        try:
            import llama_cpp # type: ignore[import-not-found,unused-ignore]
            llama_cpp # silence importchecker
            llama_cpp_available._available = True # type: ignore[attr-defined]
        except:
            llama_cpp_available._available = False # type: ignore[attr-defined]
    return llama_cpp_available._available # type: ignore[attr-defined, no-any-return]
