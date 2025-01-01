from __future__ import annotations

import sys

from typing import Any, Dict, Optional, Type, List, Callable, TypeGuard, Sequence, TYPE_CHECKING
from typing_extensions import TypedDict, NotRequired

from ..constants import NOTSET
from .log_util import logger

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image

__all__ = [
    "CallSignatureParameter",
    "CallSignature",
    "IntrospectableMixin",
    "get_signature",
    "realize_kwargs",
    "get_options_from_literal",
    "get_parameter_enum",
    "maybe_elaborate_type",
    "has_type_name",
    "is_numpy_array",
    "is_torch_tensor",
    "is_pil_image",
    "is_multiple",
    "validate_parameters",
]

class CallSignatureParameter(TypedDict):
    """
    A parameter of a call signature
    """
    parameter_type: type
    required: bool
    default: Any
    description: NotRequired[str]

class CallSignature(TypedDict):
    """
    Model of a task that can be executed by a server.
    """
    parameters: Dict[str, CallSignatureParameter]
    return_type: Optional[Type[Any]]
    short_description: NotRequired[str]
    long_description: NotRequired[str]

def is_indexable(maybe_indexable: Any) -> bool:
    """
    Check if an object is indexable.
    """
    return hasattr(maybe_indexable, "__getitem__")

def get_signature(method: Callable[..., Any]) -> CallSignature:
    """
    Get the signature of a method.

    Combines runtime introspection via `inspect` with
    docstring parsing via `docstring_parser`.
    """
    import inspect
    from docstring_parser import parse as parse_docstring
    from docstring_parser.common import DocstringParam

    signature = inspect.signature(method)
    docstring = parse_docstring(method.__doc__ or "")
    parameters: Dict[str, CallSignatureParameter] = {}

    for parameter_name in signature.parameters:
        if signature.parameters[parameter_name].kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # Skip variadic arguments (e.g. *args, **kwargs)
            continue

        param_docstring: Optional[DocstringParam] = None

        for param in docstring.params:
            if param.arg_name == parameter_name:
                param_docstring = param
                break

        parameter_default = signature.parameters[parameter_name].default
        if parameter_default is inspect._empty:
            parameter_default = NOTSET

        parameter_type = signature.parameters[parameter_name].annotation
        if parameter_type is inspect._empty:
            if param_docstring is not None and param_docstring.type_name is not None:
                parameter_type = param_docstring.type_name
            else:
                parameter_type = Any

        parameters[parameter_name] = {
            "parameter_type": maybe_elaborate_type(parameter_type),
            "required": parameter_default is NOTSET,
            "default": parameter_default,
        }

        if (
            isinstance(parameters[parameter_name]["parameter_type"], str) and
            parameters[parameter_name]["parameter_type"].startswith("Optional") # type: ignore[attr-defined]
        ):
            parameters[parameter_name]["required"] = False
            parameters[parameter_name]["parameter_type"] = maybe_elaborate_type(parameters[parameter_name]["parameter_type"][9:-1]) # type: ignore[index]

        if param_docstring is not None and param_docstring.description is not None:
            parameters[parameter_name]["description"] = param_docstring.description

    return_type = signature.return_annotation
    if return_type is inspect._empty:
        if docstring.returns is not None and docstring.returns.type_name is not None:
            return_type = docstring.returns.type_name
        else:
            return_type = Any

    signature_dict: CallSignature = {
        "parameters": parameters,
        "return_type": maybe_elaborate_type(return_type),
    }

    if docstring.short_description:
        signature_dict["short_description"] = docstring.short_description
    if docstring.long_description:
        signature_dict["long_description"] = docstring.long_description

    return signature_dict

def realize_kwargs(
    method: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Takes a method and a set of arguments and keyword arguments,
    and returns them as a dictionary of keyword arguments - i.e.,
    all positional arguments are converted to their corresponding
    keyword argument names.
    """
    signature = get_signature(method)
    parameters = signature["parameters"]
    parameter_names = list(parameters.keys())
    num_parameters = len(parameters)
    realized_kwargs: Dict[str, Any] = {}

    for i, arg in enumerate(args):
        if i >= num_parameters:
            break
        parameter_name = parameter_names[i]
        realized_kwargs[parameter_name] = arg

    for parameter_name, parameter_info in parameters.items():
        if parameter_name in kwargs:
            realized_kwargs[parameter_name] = kwargs[parameter_name]

    return realized_kwargs

class IntrospectableMixin:
    """
    Mixin for callable classes that can be introspected.
    """
    @classmethod
    def introspect(cls) -> CallSignature:
        """
        Introspect the class.
        """
        signature = get_signature(cls.__call__)
        if "self" in signature["parameters"]:
            del signature["parameters"]["self"]
        return signature

    @property
    def signature(self) -> CallSignature:
        """
        Get the signature of the class.
        """
        if not hasattr(self, "_signature"):
            self._signature = self.introspect()
        return self._signature

def get_options_from_literal(maybe_literal: Any) -> Optional[List[str]]:
    """
    Get the options from a literal type.
    """
    if hasattr(maybe_literal, "__args__") and getattr(maybe_literal, "__name__", None) == "Literal":
        return maybe_literal.__args__ # type: ignore[no-any-return]
    if isinstance(maybe_literal, str):
        if maybe_literal.startswith("Literal["):
            options = maybe_literal[8:-1].split(", ")
            return [option.strip("'\"") for option in options]
        elif maybe_literal.startswith("typing.Literal["):
            options = maybe_literal[15:-1].split(", ")
            return [option.strip("'\"") for option in options]
        elif maybe_literal.startswith("typing_extensions.Literal["):
            options = maybe_literal[26:-1].split(", ")
            return [option.strip("'\"") for option in options]
        elif maybe_literal in globals():
            return get_options_from_literal(globals()[maybe_literal])
        elif maybe_literal in locals():
            return get_options_from_literal(locals()[maybe_literal])
    return None

def get_parameter_enum(maybe_literal_or_enum: Any) -> Optional[List[str]]:
    """
    Get the options from a literal type or an enum.
    """
    options = get_options_from_literal(maybe_literal_or_enum)
    if options is not None:
        return options
    if hasattr(maybe_literal_or_enum, "__members__"):
        return list(maybe_literal_or_enum.__members__.keys())
    return None

def maybe_elaborate_type(maybe_type: Any, string: bool=False) -> Any:
    """
    Elaborate a type if it is a literal type.
    This is only relevant when reading annotations that use an imported
    literal type. This method will hopefully resolve the imported literal
    and expand on it for reporting at runtime.
    """
    import taproot
    import typing
    literal_options = get_options_from_literal(maybe_type)
    if literal_options is not None:
        return f"Literal[{', '.join(literal_options)}]"
    elif isinstance(maybe_type, str):
        if maybe_type.startswith("Optional["):
            optional_type = maybe_elaborate_type(maybe_type[9:-1], string=string)
            return f"Optional[{optional_type}]"
        elif maybe_type.startswith("Union["):
            union_types = ", ".join([
                maybe_elaborate_type(option, string=True)
                for option in maybe_type[6:-1].split(', ')
            ])
            return f"Union[{union_types}]"
        elif maybe_type.startswith("<class '"):
            return maybe_elaborate_type(maybe_type[8:-2], string=string)
        elif maybe_type == "Any":
            if string:
                return "Any"
            return Any
        elif hasattr(typing, maybe_type):
            return str(maybe_type)
        elif hasattr(taproot, maybe_type):
            return maybe_elaborate_type(getattr(taproot, maybe_type), string=string)

        maybe_type_path = maybe_type.split(".")

        for check_parent_object in [__builtins__, sys.modules, globals(), locals()]:
            maybe_return_type = check_parent_object
            for maybe_type_name in maybe_type_path:
                if hasattr(maybe_return_type, maybe_type_name):
                    maybe_return_type = getattr(maybe_return_type, maybe_type_name)
                elif is_indexable(maybe_return_type) and maybe_type_name in maybe_return_type: # type: ignore[operator]
                    maybe_return_type = maybe_return_type[maybe_type_name] # type: ignore[index]
                else:
                    maybe_return_type = None
                    break
            if maybe_return_type is not None:
                return maybe_elaborate_type(maybe_return_type, string=string)

    elif str(maybe_type).startswith("typing."):
        return maybe_elaborate_type(str(maybe_type)[7:], string=string)
    elif str(maybe_type).startswith("typing_extensions."):
        return maybe_elaborate_type(str(maybe_type)[19:], string=string)

    if not isinstance(maybe_type, str) and string:
        return getattr(maybe_type, "__name__", str(maybe_type))

    return maybe_type

MRO_TYPE_NAME_CACHE: Dict[Type[Any], List[str]] = {}
def has_type_name(maybe_type: Any, type_name: str) -> bool:
    """
    Check if a type has a specific type name.
    This allows us to check if a type is a subclass of a
    specific type without having to import the type.
    """
    global MRO_TYPE_NAME_CACHE
    if not isinstance(maybe_type, type):
        maybe_type = type(maybe_type)
    if maybe_type not in MRO_TYPE_NAME_CACHE:
        MRO_TYPE_NAME_CACHE[maybe_type] = [
            mro_type.__name__.lower() for mro_type in maybe_type.mro()
        ]
    return type_name.lower() in MRO_TYPE_NAME_CACHE[maybe_type]

def is_numpy_array(maybe_array: Any) -> TypeGuard[np.ndarray[Any,Any]]:
    """
    Check if an object is a numpy array.
    """
    return has_type_name(maybe_array, "ndarray")

def is_torch_tensor(maybe_tensor: Any) -> TypeGuard[torch.Tensor]:
    """
    Check if an object is a torch tensor.
    """
    return has_type_name(maybe_tensor, "tensor")

def is_pil_image(maybe_image: Any) -> TypeGuard[Image.Image]:
    """
    Check if an object is a PIL image.
    """
    return has_type_name(maybe_image, "image")

def is_multiple(maybe_list: Any) -> bool:
    """
    Check if an object is a list of elements.
    """
    if (
        isinstance(maybe_list, list) or
        isinstance(maybe_list, tuple) or
        isinstance(maybe_list, set)
    ):
        return len(maybe_list) > 1
    return False

def validate_parameters(
    method_parameters: Dict[str, CallSignatureParameter],
    invoke_parameters: Dict[str, Any],
    include_defaults: bool=True,
    raise_on_missing: bool=True,
    raise_on_invalid: bool=True,
    raise_on_extra: bool=False,
) -> Dict[str, Any]:
    """
    Validate the parameters of a method.
    """
    validated_parameters = {}
    ignored_parameters = []

    for parameter_name, parameter_info in method_parameters.items():
        if parameter_name in invoke_parameters:
            parameter_type = parameter_info["parameter_type"]
            maybe_enum = get_parameter_enum(parameter_type)

            if maybe_enum is not None and invoke_parameters[parameter_name] not in maybe_enum:
                if raise_on_invalid:
                    raise ValueError(f"Parameter '{parameter_name}' must be one of {maybe_enum}")
                continue
            elif maybe_enum is None and isinstance(parameter_type, type):
                if not isinstance(invoke_parameters[parameter_name], parameter_type):
                    # Try to cast
                    try:
                        invoke_parameters[parameter_name] = parameter_type(invoke_parameters[parameter_name])
                    except Exception as e:
                        if raise_on_invalid:
                            raise TypeError(f"Parameter '{parameter_name}' must be of type {parameter_type.__name__}") from e
                        else:
                            logger.warning(f"Parameter '{parameter_name}' must be of type {parameter_type.__name__}")
                            continue
            validated_parameters[parameter_name] = invoke_parameters[parameter_name]
        elif parameter_info["required"] and raise_on_missing:
            raise TypeError(f"Parameter '{parameter_name}' is required")
        elif "default" in parameter_info and include_defaults:
            validated_parameters[parameter_name] = parameter_info["default"]

    for parameter_name in invoke_parameters:
        if parameter_name not in method_parameters:
            if raise_on_extra:
                raise TypeError(f"Unexpected parameter '{parameter_name}'")
            else:
                ignored_parameters.append(parameter_name)

    if ignored_parameters:
        logger.warning(f"Ignored parameters: {ignored_parameters}")

    return validated_parameters
