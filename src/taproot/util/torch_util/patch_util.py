from __future__ import annotations

from typing import Iterator, Any, Type, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    import torch

__all__ = ["inject_skip_init"]

@contextmanager
def inject_skip_init(*modules: Type[torch.nn.Module]) -> Iterator[None]:
    """
    Context manager that does the same thing as `torch.nn.utils.init.skip_init`, but hijacks the passed
    modules __init__ method to perform the operations instead of requiring the user to call `skip_init` on
    the module instance.

    This should only be used in special circumstances.
    """
    import torch

    for module in modules:
        assert issubclass(module, torch.nn.Module), f"Expected a subclass of torch.nn.Module, got {module}"
        module_init = module.__init__
        setattr(module, "__original_init__", module_init)

        def new_module_init(self: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
            final_device = kwargs.pop("device", "cpu")
            kwargs["device"] = "meta"
            module_init(self, *args, **kwargs)
            self.to_empty(device=final_device)

        module.__init__ = new_module_init # type: ignore[method-assign]

    try:
        yield
    finally:
        for module in modules:
            module.__init__ = module.__original_init__ # type: ignore[method-assign,attr-defined]
            delattr(module, "__original_init__")
