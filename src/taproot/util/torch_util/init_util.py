from typing import Any, Iterator
from contextlib import contextmanager

__all__ = ["no_init_weights"]

@contextmanager
def no_init_weights() -> Iterator[None]:
    """
    Context manager to globally disable weight initialization to speed up loading large models.
    """
    try:
        from transformers.modeling_utils import no_init_weights # type: ignore[import-untyped,import-not-found,unused-ignore]
        with no_init_weights():
            yield
    except ImportError:
        import torch
        torch_init_functions = {
            "uniform_": torch.nn.init.uniform_,
            "normal_": torch.nn.init.normal_,
            "trunc_normal_": torch.nn.init.trunc_normal_,
            "constant_": torch.nn.init.constant_,
            "xavier_uniform_": torch.nn.init.xavier_uniform_,
            "xavier_normal_": torch.nn.init.xavier_normal_,
            "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
            "kaiming_normal_": torch.nn.init.kaiming_normal_,
            "uniform": torch.nn.init.uniform,
            "normal": torch.nn.init.normal,
            "xavier_uniform": torch.nn.init.xavier_uniform,
            "xavier_normal": torch.nn.init.xavier_normal,
            "kaiming_uniform": torch.nn.init.kaiming_uniform,
            "kaiming_normal": torch.nn.init.kaiming_normal,
        }

        def _skip_init(*args: Any, **kwargs: Any) -> None:
            pass

        for name in torch_init_functions.keys():
            setattr(torch.nn.init, name, _skip_init)
        try:
            yield
        finally:
            # Restore the original initialization functions
            for name, init_func in torch_init_functions.items():
                setattr(torch.nn.init, name, init_func)
