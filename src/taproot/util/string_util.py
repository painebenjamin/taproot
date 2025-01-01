from __future__ import annotations

import re

from uuid import uuid4
from math import modf
from random import choices
from typing import Tuple, Union, Optional, Sequence

__all__ = [
    "get_uuid",
    "reduce_units",
    "random_ascii_string",
    "human_size",
    "human_duration",
    "trim_docstring",
    "indent_docstring",
    "simplify_quotations",
    "multiline_trim",
]

ASCII_CHARS = [
    chr(code)
    for code in range(32, 127) # ASCII
]

def get_uuid() -> str:
    """
    Generate a random UUID.
    """
    return uuid4().hex

def random_ascii_string(length: int) -> str:
    """
    Generate a random ASCII string of a given length.

    >>> len(random_ascii_string(10))
    10
    """
    return "".join(choices(ASCII_CHARS, k=length))

def reduce_units(
    value: Union[int, float],
    units: Sequence[Union[str, Tuple[str, Union[int, float]]]],
    base: Union[int, float] = 1000,
) -> Tuple[float, str]:
    """
    Reduce a value to the smallest unit possible.

    >>> reduce_units(4e9, ["bytes/s", "kb/s", "mb/s", "gb/s"])
    (4.0, 'gb/s')
    """
    try:
        unit = units[0]
    except IndexError:
        raise ValueError("At least one unit must be provided.")

    for unit_or_tuple in units:
        if isinstance(unit_or_tuple, tuple):
            unit, unit_base = unit_or_tuple
        else:
            unit = unit_or_tuple
            unit_base = base
        if value < unit_base:
            break
        value /= unit_base
    return value, unit # type: ignore[return-value]

def human_size(
    num_bytes: Union[int, float],
    base_2: bool = False,
    precision: int = 2
) -> str:
    """
    Convert a number of bytes to a human-readable string.

    >>> human_size(1000)
    '1.00 KB'
    >>> human_size(1000**3)
    '1.00 GB'
    >>> human_size(1024, base_2=True)
    '1.00 KiB'
    >>> human_size(1024**3, base_2=True)
    '1.00 GiB'
    """
    if base_2:
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        divisor = 1024.0
    else:
        units = ["B", "KB", "MB", "GB", "TB"]
        divisor = 1000.0

    reduced_bytes, unit = reduce_units(num_bytes, units, base=divisor)

    return f"{reduced_bytes:.{precision}f} {unit}"

def human_duration(
    duration_s: Union[int, float],
    precision: Optional[float] = None,
) -> str:
    """
    Convert a number of seconds to a human-readable string.
    Decimal precision is variable.

    Value < 1 second:
        Nanoseconds, microseconds, and milliseconds are reported as integers.
    1 second < value < 1 minute:
        Seconds are reported as floats with one decimal place.
    1 minute < value < 1 hour:
        Reported as minutes and seconds in the format "<x> m <y> s" with no decimal places.
    1 hour < value < 1 day:
        Reported as hours and minutes in the format "<x> h <y> m <z> s" with no decimal places.
    1 day < value:
        Reported as days and hours in the format "<x> d <y> h <z> m <zz> s" with no decimal places.

    >>> human_duration(0.00001601)
    '16 µs'
    >>> human_duration(1.5)
    '1.5 s'
    >>> human_duration(65)
    '1 m 5 s'
    >>> human_duration(3665)
    '1 h 1 m 5 s'
    >>> human_duration(90065)
    '1 d 1 h 1 m 5 s'
    """
    # First set the duration to nanoseconds
    duration_s *= 1e9
    units = ["ns", "µs", "ms", "s", "m", "h", "d"]
    bases = [1e3, 1e3, 1e3, 60, 60, 24, 1000]
    reduced_seconds, unit = reduce_units(
        duration_s,
        list(zip(units, bases)),
        base=1000,
    )
    if unit in ["d", "h", "m"]:
        # Split the seconds into a whole part and a fractional part
        fractional, whole = modf(reduced_seconds)
        whole_formatted = f"{whole:.0f} {unit}"
        if fractional == 0:
            return whole_formatted
        # Return the fractional part to seconds
        if unit in ["d", "h", "m"]:
            fractional *= 60
        if unit in ["d", "h"]:
            fractional *= 60
        if unit == "d":
            fractional *= 24
        return " ".join([
            whole_formatted,
            human_duration(fractional, precision=0)
        ])
    else:
        if unit in ["ns", "µs", "ms"] and precision is None:
            precision = 1 if reduced_seconds < 10 else 0
        elif unit == "s" and precision is None:
            precision = 1
        return f"{reduced_seconds:.{precision}f} {unit}"

def trim_docstring(text: str) -> str:
    """
    Trim leading and trailing whitespace from each paragraph in a string.
    """
    # Split the string into lines
    lines = text.split('\n')

    # Remove leading and trailing blank lines
    while lines and lines[0].strip() == '':
        lines.pop(0)
    while lines and lines[-1].strip() == '':
        lines.pop()

    if not lines:
        return ""

    # Find the minimum indentation (ignoring empty lines)
    min_indent = float('inf')
    for line in lines:
        stripped_line = line.lstrip()
        if stripped_line:
            indent = len(line) - len(stripped_line)
            if indent < min_indent:
                min_indent = indent

    # Remove the minimum indentation from all lines
    trimmed_lines = [line[min_indent:] if line.strip() else '' for line in lines] # type: ignore[misc]

    # Join the lines back into a single string
    return '\n'.join(trimmed_lines)

def indent_docstring(text: str, indent: int=2) -> str:
    """
    Indent each line in a string by a given number of spaces.
    """
    trimmed = trim_docstring(text)
    return '\n'.join([' ' * indent + line for line in trimmed.split('\n')])

single_quote_regex = re.compile(r"[‚‘’′‵`‛]")
double_quote_regex = re.compile(r"[„“”″‶″‴〃‷]")

def simplify_quotations(text: str) -> str:
    """
    Simplify the quotation marks in a string - for example, turning
    angled quotes into straight quotes. Applies to both single and
    double quote marks.
    """
    text = single_quote_regex.sub("'", text)
    text = double_quote_regex.sub('"', text)
    return text

def multiline_trim(text: str) -> str:
    """
    Performs the following operations on a multiline string:
    1. Replaces contiguous empty (only whitespace) lines with a single empty line
    2. Replaces contiguous spaces with a single space
    3. Removes all leading and trailing whitespace
    """
    text = re.sub(r'(\n\s*\n)+', '\n\n', text)
    text = re.sub(r'\ +', ' ', text)
    return text.strip()
