from __future__ import annotations

from typing import Dict, Type, Union, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import dtype as DType

__all__ = [
    "get_torch_dtype",
    "get_num_bits_for_dtype",
    "is_8_bit_dtype",
    "is_16_bit_dtype",
    "is_32_bit_dtype",
    "is_64_bit_dtype",
    "is_128_bit_dtype",
    "TorchDataTypeConverter",
]

class TorchDataTypeConverter:
    """
    This class converts between numpy and torch types.
    Also provides helper functions for converting from strings.
    """

    @classmethod
    def from_string(cls, torch_type: str) -> DType:
        """
        Converts a string to a torch DType.
        """
        import torch
        try:
            return {
                "complex128": torch.complex128,
                "cdouble": torch.complex128,
                "complex": torch.complex64,
                "complex64": torch.complex64,
                "cfloat": torch.complex64,
                "cfloat64": torch.complex64,
                "cf64": torch.complex64,
                "double": torch.float64,
                "float64": torch.float64,
                "fp64": torch.float64,
                "float": torch.float32,
                "full": torch.float32,
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "fp8": torch.float8_e4m3fn,
                "float8": torch.float8_e4m3fn,
                "float8_e4m3": torch.float8_e4m3fn,
                "float8_e4m3fn": torch.float8_e4m3fn,
                "fp84": torch.float8_e4m3fn,
                "float8_e5m2": torch.float8_e5m2,
                "fp85": torch.float8_e5m2,
                "uint8": torch.uint8,
                "int8": torch.int8,
                "int16": torch.int16,
                "short": torch.int16,
                "int": torch.int32,
                "int32": torch.int32,
                "long": torch.int64,
                "int64": torch.int64,
                "bool": torch.bool,
                "bit": torch.bool,
                "1": torch.bool
            }[torch_type[6:] if torch_type.startswith("torch.") else torch_type]
        except KeyError:
            raise ValueError(f"Unknown torch type '{torch_type}'")

    @classmethod
    def from_torch(cls, torch_type: DType) -> Type[Any]:
        """
        Gets the numpy type from torch.
        :raises: KeyError When type is unknown.
        """
        import torch
        import numpy as np
        torch_to_numpy: Dict[torch.dtype, Type[Any]] = {
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.complex64: np.complex64,
            torch.complex128: np.complex128,
            torch.bool: np.bool_,
        }
        return torch_to_numpy[torch_type]

    @classmethod
    def from_numpy(cls, numpy_type: Type[Any]) -> DType:
        """
        Gets the torch type from nump.
        :raises: KeyError When type is unknown.
        """
        import torch
        import numpy as np
        numpy_to_torch: Dict[Type[Any], torch.dtype] = {
            np.uint8: torch.uint8,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.complex64: torch.complex64,
            np.complex128: torch.complex128,
            np.bool_: torch.bool,
        }
        return numpy_to_torch[numpy_type]

    @classmethod
    def convert(cls, type_to_convert: Optional[Union[DType, Type[Any], str]]) -> Optional[DType]:
        """
        Converts to a torch DType
        """
        import torch
        if type_to_convert is None:
            return None
        if isinstance(type_to_convert, torch.dtype):
            return type_to_convert
        if isinstance(type_to_convert, str):
            return cls.from_string(str(type_to_convert)) # Raises
        try:
            return cls.from_numpy(type_to_convert)
        except KeyError:
            return cls.from_string(str(type_to_convert)) # Raises

def get_torch_dtype(dtype: Union[DType, Type[Any], str]) -> DType:
    """
    Gets the torch data type from a string.
    """
    converted_dtype = TorchDataTypeConverter.convert(dtype)
    if converted_dtype is None:
        raise ValueError(f"Could not convert '{dtype}' to torch dtype.")
    return converted_dtype

def get_num_bits_for_dtype(dtype: DType) -> int:
    """
    Gets the number of bits for a torch dtype.
    """
    import torch
    return torch.finfo(dtype).bits

def is_8_bit_dtype(dtype: DType) -> bool:
    """
    Checks if the dtype is 8 bits.
    """
    return get_num_bits_for_dtype(dtype) == 8

def is_16_bit_dtype(dtype: DType) -> bool:
    """
    Checks if the dtype is 16 bits.
    """
    return get_num_bits_for_dtype(dtype) == 16

def is_32_bit_dtype(dtype: DType) -> bool:
    """
    Checks if the dtype is 32 bits.
    """
    return get_num_bits_for_dtype(dtype) == 32

def is_64_bit_dtype(dtype: DType) -> bool:
    """
    Checks if the dtype is 64 bits.
    """
    return get_num_bits_for_dtype(dtype) == 64

def is_128_bit_dtype(dtype: DType) -> bool:
    """
    Checks if the dtype is 128 bits.
    """
    return get_num_bits_for_dtype(dtype) == 128
