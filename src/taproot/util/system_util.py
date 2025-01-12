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
def catch_output(
    ignore_output: bool=False,
    reset_on_exit: bool=True
) -> Iterator[SystemOutputCatcher]:
    """
    A context manager that allows easy capturing of stdout/stderr

    >>> with catch_output() as catcher:
    ...     print("stdout")
    >>> catcher.out
    'stdout'

    :param ignore_output: If True, output will be ignored.
    :param reset_on_exit: If True, the system streams will
        be reset to their original values when exiting the context.
    """
    catcher = SystemOutputCatcher(
        ignore_output=ignore_output,
        reset_on_exit=reset_on_exit
    )
    with catcher:
        yield catcher

class DummyStandardOutput(StringIO):
    """
    A StringIO that acts like a standard output stream.
    """
    def fileno(self) -> int:
        """
        1 is the file descriptor for stdout.
        """
        return 1

class DummyStandardError(StringIO):
    """
    A StringIO that acts like a standard error stream.
    """
    def fileno(self) -> int:
        """
        2 is the file descriptor for stderr.
        """
        return 2

class BlackHole(StringIO):
    """
    A StringIO that doesn't store anything.
    """
    def write(self, s: str) -> int:
        """
        Writes nothing (but lies about it).
        """
        return len(s)

class BlackHoleStandardOutput(DummyStandardOutput, BlackHole): # type: ignore[misc]
    """
    A StringIO that acts like a standard output stream, but doesn't store anything.
    """
    pass

class BlackHoleStandardError(DummyStandardError, BlackHole): # type: ignore[misc]
    """
    A StringIO that acts like a standard error stream, but doesn't store anything.
    """
    pass

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
    stdout: StringIO
    stderr: StringIO

    def __init__(
        self,
        reset_on_exit: bool=True,
        ignore_output: bool=False
    ) -> None:
        """
        Initialize IOs for stdout and stderr.
        """
        self.reset_on_exit = reset_on_exit
        self.ignore_output = ignore_output
        if ignore_output:
            self.stdout = BlackHoleStandardOutput()
            self.stderr = BlackHoleStandardError()
        else:
            self.stdout = DummyStandardOutput()
            self.stderr = DummyStandardError()

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
        if self.reset_on_exit:
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
        if not self.ignore_output:
            self.stdout = DummyStandardOutput()
            self.stderr = DummyStandardError()

    def output(self) -> Tuple[str, str]:
        """
        Returns the contents of stdout and stderr.
        """
        return (self.out, self.err)
