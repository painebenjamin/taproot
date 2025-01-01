from __future__ import annotations

from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "wrap_module_forward_dtype",
    "unwrap_module_forward_dtype"
]

def wrap_module_forward_dtype(
    module: torch.nn.Module,
    input_dtype: Optional[torch.dtype] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> None:
    """
    Wrap a module to cast input and output tensors to the given dtypes.
    """
    import torch

    if hasattr(module, "_pre_wrap_module_forward_dtype"):
        module_forward = module._pre_wrap_module_forward_dtype
    else:
        module_forward = module.forward

    def maybe_convert(arg: Any, dtype: torch.dtype) -> Any:
        """
        Converts the arg if it is a tensor.
        """
        if isinstance(arg, torch.Tensor) and arg.is_floating_point():
            return arg.to(dtype=dtype)
        elif isinstance(arg, (list, tuple)):
            return type(arg)(maybe_convert(a, dtype) for a in arg)
        elif isinstance(arg, dict):
            return {k: maybe_convert(v, dtype) for k, v in arg.items()}
        return arg

    def wrap_forward(*args: Any, **kwargs: Any) -> Any:
        """
        Wrap the forward method of the module.
        """
        if input_dtype is not None:
            args = maybe_convert(args, input_dtype)
            kwargs = maybe_convert(kwargs, input_dtype)

        output = module_forward(*args, **kwargs)

        if output_dtype is not None:
            output = maybe_convert(output, output_dtype)

        return output

    setattr(module, "_pre_wrap_module_forward_dtype", module_forward)
    module.forward = wrap_forward

def unwrap_module_forward_dtype(module: torch.nn.Module) -> None:
    """
    Unwrap a module that was wrapped by `wrap_module_forward_dtype`.
    """
    if hasattr(module, "_pre_wrap_module_forward_dtype"):
        module.forward = module._pre_wrap_module_forward_dtype
        del module._pre_wrap_module_forward_dtype
