from __future__ import annotations

import inspect

from typing import Optional, Any, TYPE_CHECKING
from typing_extensions import Literal

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "ACTIVATION_FUNCTION_LITERAL",
    "get_activation"
]

ACTIVATION_FUNCTION_LITERAL = Literal["relu", "leaky_relu", "gelu", "silu", "swish", "mish", "tanh", "sigmoid", "identity"]

def get_activation(
    activation: Optional[ACTIVATION_FUNCTION_LITERAL] = None,
    **kwargs: Any
) -> nn.Module:
    """
    Returns an activation function module based on the provided name.

    The supported activation functions are:

    - "relu": Rectified Linear Unit, commonly used in many neural networks.
    - "leaky_relu": Leaky version of a Rectified Linear Unit with a small slope for negative values.
    - "gelu": Gaussian Error Linear Unit, often used in transformer architectures.
    - "silu": Sigmoid Linear Unit, also known as "swish", a smooth non-linear activation.
    - "swish": Alias for "silu" for consistency with other naming conventions.
    - "mish": Another smooth activation function similar to "swish".
    - "tanh": Hyperbolic tangent, used in tasks requiring a bounded output in the range [-1, 1].
    - "sigmoid": Produces outputs in the range [0, 1], useful for binary classification tasks.
    - "identity": No-op activation, returns the input unchanged. This can be useful when you want
      to effectively "turn off" activation without modifying the model structure.

    :param activation: A string representing the name of the desired activation function.
                       If None is provided, or if "identity" is selected, no activation will be applied.
    :param args: Additional positional arguments to be passed to the activation function, if any.
    :param kwargs: Additional keyword arguments to be passed to the activation function, if any.
    :return: A PyTorch activation function module (`torch.nn.Module`).
    :raises ValueError: If an unknown activation function is passed.
    """
    import torch.nn as nn
    activation_map = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "mish": nn.Mish,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
        None: nn.Identity
    }.get(activation, None)
    if activation_map is None:
        raise ValueError(f"Activation function '{activation}' not found.")
    accepted_kwargs = inspect.signature(activation_map).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in accepted_kwargs}
    return activation_map(**kwargs) # type: ignore[no-any-return]
