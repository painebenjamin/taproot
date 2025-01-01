import time

from typing import Any, Optional

from .base import Task
from ..constants import *

__all__ = ["Echo"]

class Echo(Task):
    """
    A task for responding with the same message sent to it.
    Useful for testing connectivity in production.
    """
    task: str = "echo"
    default: bool = True

    """Authorship Metadata"""
    author = "Benjamin Paine"
    author_url = "https://github.com/painebenjamin/taproot"
    author_affiliations = ["Taproot"]

    """License Metadata"""
    license = LICENSE_APACHE

    def __call__( # type: ignore[override]
        self,
        *,
        message: Any,
        delay: Optional[float]=None
    ) -> Any:
        """
        Returns the same message passed to it, optionally with a delay.
        """
        if delay is not None and delay > 0:
            time.sleep(delay)
        return message
