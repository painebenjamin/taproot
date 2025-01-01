from __future__ import annotations

import os
import sys
import platform
import subprocess

from io import StringIO
from contextlib import contextmanager

from typing import Any, Tuple, Iterator

__all__ = [
    "open_file",
    "catch_output"
]

def open_file(path: str) -> None:
    """
    Opens a file using the system default program for that file type.
    """
    system = platform.system()
    if system == "Windows":
        os.startfile(path) # type: ignore[attr-defined,unused-ignore]
    elif system == "Darwin":
        subprocess.Popen(["open", path])
    elif system == "Linux":
        subprocess.Popen(["xdg-open", path])
    else:
        raise OSError(f"Unsupported operating system: {system}")

@contextmanager
def catch_output() -> Iterator[SystemOutputCatcher]:
    """
    A context manager that allows easy capturing of stdout/stderr

    >>> with catch_output() as catcher:
    ...     print("stdout")
    >>> catcher.out
    'stdout'
    """
    catcher = SystemOutputCatcher()
    with catcher:
        yield catcher

class DummyFileStringIO(StringIO):
    """
    A StringIO that doesn't break when fileno() is called.
    """
    def fileno(self) -> int:
        """
        Returns 0, since this is not a real file.
        """
        return 0

class SystemOutputCatcher:
    """
    A context manager that allows easy capturing of stdout/stderr

    >>> catcher = SystemOutputCatcher()
    >>> catcher.__enter__()
    >>> print("stdout")
    >>> catcher.__exit__()
    >>> catcher.output()[0].strip()
    'stdout'
    """

    def __init__(self) -> None:
        """
        Initialize IOs for stdout and stderr.
        """
        self.stdout = DummyFileStringIO()
        self.stderr = DummyFileStringIO()

    def __enter__(self) -> None:
        """
        When entering context, steal system streams.
        """
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr

    def __exit__(self, *args: Any) -> None:
        """
        When exiting context, return system streams.
        """
        if hasattr(self, "_stdout"):
            sys.stdout = self._stdout
        if hasattr(self, "_stderr"):
            sys.stderr = self._stderr

    @property
    def out(self) -> str:
        """
        Returns the contents of stdout.
        """
        return self.stdout.getvalue()

    @property
    def err(self) -> str:
        """
        Returns the contents of stderr.
        """
        return self.stderr.getvalue()

    def clean(self) -> None:
        """
        Cleans memory by replacing StringIO.
        This is faster than trunc/seek
        """
        self.stdout = DummyFileStringIO()
        self.stderr = DummyFileStringIO()

    def output(self) -> Tuple[str, str]:
        """
        Returns the contents of stdout and stderr.
        """
        return (self.out, self.err)
