__all__ = [
    "NotCapableError",
    "CapacityExceededError",
    "CapacityReservedError"
]

class NotCapableError(Exception):
    """
    Exception raised when an object is not capable of performing the operation.
    """
    def __init__(
        self,
        message: str = "Not capable of performing the operation."
    ) -> None:
        super().__init__(message)

class CapacityExceededError(Exception):
    """
    Exception raised when the capacity of a queue/server is exceeded.
    """
    def __init__(
        self,
        message: str = "Capacity exceeded."
    ) -> None:
        super().__init__(message)

class CapacityReservedError(Exception):
    """
    Exception raised when the capacity of a queue/server is reserved.
    """
    def __init__(
        self,
        message: str = "Capacity reserved."
    ) -> None:
        super().__init__(message)
