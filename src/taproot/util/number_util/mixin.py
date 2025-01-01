from typing import Any, Union

class NumericMixin:
    """
    Mixin class to proxy numeric operations to a custom value.
    """
    @property
    def numeric(self) -> Union[int, float, complex]:
        """
        Override this method to return the value to be proxied.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __add__(self, other: Any) -> Any:
        """
        Proxy addition to the numeric value.
        """
        return self.numeric + other

    def __radd__(self, other: Any) -> Any:
        """
        Proxy right addition to the numeric value.
        """
        return other + self.numeric

    def __sub__(self, other: Any) -> Any:
        """
        Proxy subtraction to the numeric value.
        """
        return self.numeric - other

    def __rsub__(self, other: Any) -> Any:
        """
        Proxy right subtraction to the numeric value.
        """
        return other - self.numeric

    def __mul__(self, other: Any) -> Any:
        """
        Proxy multiplication to the numeric value.
        """
        return self.numeric * other

    def __rmul__(self, other: Any) -> Any:
        """
        Proxy right multiplication to the numeric value.
        """
        return other * self.numeric

    def __truediv__(self, other: Any) -> Any:
        """
        Proxy true division to the numeric value.
        """
        return self.numeric / other

    def __rtruediv__(self, other: Any) -> Any:
        """
        Proxy right true division to the numeric value.
        """
        return other / self.numeric

    def __floordiv__(self, other: Any) -> Any:
        """
        Proxy floor division to the numeric value.
        """
        return self.numeric // other

    def __rfloordiv__(self, other: Any) -> Any:
        """
        Proxy right floor division to the numeric value.
        """
        return other // self.numeric

    def __mod__(self, other: Any) -> Any:
        """
        Proxy modulo to the numeric value.
        """
        return self.numeric % other

    def __rmod__(self, other: Any) -> Any:
        """
        Proxy right modulo to the numeric value.
        """
        return other % self.numeric

    def __pow__(self, other: Any, modulo: Any = None) -> Any:
        """
        Proxy exponentiation to the numeric value.
        """
        return pow(self.numeric, other, modulo)

    def __rpow__(self, other: Any) -> Any:
        """
        Proxy right exponentiation to the numeric value.
        """
        return pow(other, self.numeric)

    def __neg__(self) -> Any:
        """
        Proxy negation to the numeric value.
        """
        return -self.numeric

    def __pos__(self) -> Any:
        """
        Proxy positive to the numeric value.
        """
        return +self.numeric

    def __abs__(self) -> Any:
        """
        Proxy absolute value to the numeric value.
        """
        return abs(self.numeric)

    def __int__(self) -> int:
        """
        Proxy integer conversion to the numeric value.
        """
        return int(self.numeric) # type: ignore[arg-type]

    def __float__(self) -> float:
        """
        Proxy float conversion to the numeric value.
        """
        return float(self.numeric) # type: ignore[arg-type]

    def __complex__(self) -> complex:
        """
        Proxy complex conversion to the numeric value.
        """
        return complex(self.numeric)

    def __eq__(self, other: Any) -> bool:
        """
        Proxy equality to the numeric value.
        """
        return self.numeric == other # type: ignore[no-any-return]

    def __lt__(self, other: Any) -> bool:
        """
        Proxy less than to the numeric value.
        """
        return self.numeric < other # type: ignore[no-any-return]

    def __le__(self, other: Any) -> bool:
        """
        Proxy less than or equal to to the numeric value.
        """
        return self.numeric <= other # type: ignore[no-any-return]

    def __gt__(self, other: Any) -> bool:
        """
        Proxy greater than to the numeric value.
        """
        return self.numeric > other # type: ignore[no-any-return]

    def __ge__(self, other: Any) -> bool:
        """
        Proxy greater than or equal to to the numeric value.
        """
        return self.numeric >= other # type: ignore[no-any-return]

    def __str__(self) -> str:
        """
        Proxy string representation to the numeric value.
        """
        return str(self.numeric)

    def __repr__(self) -> str:
        """
        Proxy representation to the numeric value.
        """
        return str(self.numeric)
