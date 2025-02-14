from __future__ import annotations

import re
import unicodedata

from typing import Tuple, Union, Optional, Sequence, Any, List, TYPE_CHECKING

from uuid import uuid4
from math import modf
from random import choices

if TYPE_CHECKING:
    try:
        from sudachipy import Dictionary # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError:
        pass

__all__ = [
    "get_uuid",
    "reduce_units",
    "random_ascii_string",
    "human_size",
    "human_duration",
    "trim_html_whitespace",
    "trim_docstring",
    "indent_docstring",
    "simplify_quotations",
    "multiline_trim",
    "normalize_text",
    "normalize_jp_text",
    "ends_with_multi_byte_character",
    "chunk_text",
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

def trim_html_whitespace(html: str) -> str:
    """
    Trims excess whitespace in HTML.
    """
    html = re.sub(r">\s+<", "><", html)
    html = re.sub(r"\s{2,}", " ", html)
    html = re.sub(r"(?<=>)\s+|\s+(?=<)", "", html)
    return html

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

def flip_money(m: re.Match[Any]) -> str:
    """
    Flips the order of the currency and the amount.
    """
    m = m.group()
    bill = "dollar" if m[0] == "$" else "pound"
    if m[-1].isalpha():
        return f"{m[1:]} {bill}s" # type: ignore[call-overload]
    elif "." not in m: # type: ignore[operator]
        s = "" if m[1:] == "1" else "s" # type: ignore[call-overload]
        return f"{m[1:]} {bill}{s}" # type: ignore[call-overload]
    b, c = m[1:].split(".") # type: ignore[call-overload]
    s = "" if b == "1" else "s"
    c = int(c.ljust(2, "0"))
    coins = (
        f"cent{'' if c == 1 else 's'}"
        if m[0] == "$"
        else (
            "penny"
            if c == 1
            else "pence"
        )
    )
    return f"{b} {bill}{s} and {c} {coins}"

def point_num(num: re.Match[Any]) -> str:
    """
    Replaces commas with periods in decimals.
    """
    a, b = num.group().split(".")
    return " point ".join([a, " ".join(b)])

def split_num(num: re.Match[Any]) -> str:
    """
    Splits a number into its components.
    :return: The number split into its components.
    """
    num = num.group()
    if '.' in num: # type: ignore[operator]
        return num # type: ignore[return-value]
    elif ':' in num: # type: ignore[operator]
        h, m = [int(n) for n in num.split(':')] # type: ignore[attr-defined]
        if m == 0:
            return f"{h} o'clock"
        elif m < 10:
            return f'{h} oh {m}'
        return f'{h} {m}'
    year = int(num[:4]) # type: ignore[call-overload]
    if year < 1100 or year % 1000 < 10:
        return num # type: ignore[return-value]
    left, right = num[:2], int(num[2:4]) # type: ignore[call-overload]
    s = 's' if num.endswith('s') else '' # type: ignore[attr-defined]
    if 100 <= year % 1000 <= 999:
        if right == 0:
            return f'{left} hundred{s}'
        elif right < 10:
            return f'{left} oh {right}{s}'
    return f'{left} {right}{s}'

def normalize_text(
    text: str,
    flip_currency: bool=True,
    expand_degrees: bool=True,
    expand_honorifics: bool=True,
    expand_units: bool=True,
    ampersand_to_and: bool=True,
    at_symbol_to_at: bool=True,
    negative_numbers: bool=True,
    decimal_as_point: bool=True,
    split_numbers: bool=True,
) -> str:
    """
    Formats a prompt for TTS models.

    There are some quirks with how the model pronounces certain formats of text.
    This method attempts to fix some of these issues.
    """
    if negative_numbers:
        # Look for numbers that lead with a '-' and replace with 'negative'
        text = re.sub(r"([^a-zA-Z0-9])-(\d+)", r"\1negative \2", text)
    if expand_degrees:
        # Look for degrees F, replaces with 'Fahrenheit'
        text = re.sub(r"(\d+)\s*°\s*F", r"\1 degrees Fahrenheit", text)
        # Look for degrees C, replaces with 'Celsius'
        text = re.sub(r"(\d+)\s*°\s*C", r"\1 degrees Celsius", text)
    if flip_currency:
        # Look for currency symbols and flip the order of the currency and the amount
        text = re.sub(r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b", flip_money, text)
    if expand_units:
        # Replace common abbreviations
        text = re.sub(r"(\d+)\W*[mM][pP][hH]\W", r"\1 miles per hour", text)
        text = re.sub(r"(\d+)\W*[kK][pP][hH]\W", r"\1 kilometers per hour", text)
        text = re.sub(r"(\d+)\W*[kK][gG]\W", r"\1 kilograms", text)
        text = re.sub(r"(\d+)\W*[lL][bB][sS]?\W", r"\1 pounds", text)
        text = re.sub(r"(\d+)\W*[fF][tT]\W", r"\1 feet", text)
        text = re.sub(r"(\d+)\W*[mM][\W]", r"\1 meters", text)
        text = re.sub(r"(\d+)\W*[cC][mM]\W", r"\1 centimeters", text)
        text = re.sub(r"(\d+)\W*[iI][nN]\W", r"\1 inches", text)
        text = re.sub(r"(\d+)\W*[mM][iI]\W", r"\1 miles", text)
        text = re.sub(r"(\d+)\W*[kK][mM]\W", r"\1 kilometers", text)
    if expand_honorifics:
        # Replace common honorifics
        text = re.sub(r"Mr\.", "Mister", text)
        text = re.sub(r"Mrs\.", "Missus", text)
        text = re.sub(r"Ms\.", "Miss", text)
        text = re.sub(r"Dr\.", "Doctor", text)
        text = re.sub(r"Prof\.", "Professor", text)
    if at_symbol_to_at:
        # Replace the 'at' symbol with 'at'
        text = re.sub(r"@", " at ", text)
    if ampersand_to_and:
        # Replace ampersands with 'and'
        text = re.sub(r"&", " and ", text)
    if decimal_as_point:
        # Replace commas with periods in decimals
        text = re.sub(r"\d*\.\d+", point_num, text)
    if split_numbers:
        # Split numbers into their components
        text = re.sub(r"\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)", split_num, text)

    # Replace newlines and carriage returns with spaces
    text = re.sub(r"([\n\r])", " ", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove uncommon punctuation which can be interpreted as other languages
    text = re.sub(r"[^a-zA-Z0-9,.:;\ \-'\"\/()!?]", "", text)

    return text.strip()

def normalize_jp_text(text: str, dictionary: Optional[Dictionary]=None) -> str:
    """
    Normalize Japanese text.
    """
    try:
        from kanjize import number2kanji
        from sudachipy import Dictionary, SplitMode
    except ImportError as ex:
        raise ImportError("Japanese language dependencies not installed. Install the 'jp' extras set.") from ex

    if dictionary is None:
        dictionary = Dictionary(dict="full").create()

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\d+", lambda m: number2kanji(int(m[0])), text)
    text = " ".join([    
        x.reading_form()
        for x in dictionary.tokenize(text, SplitMode.A)
    ])
    return text

def ends_with_multi_byte_character(text: str) -> bool:
    """
    Check if the text ends with a multi-byte character.
    """
    return len(text[-1].encode("utf-8")) > 1

def chunk_text(text: str, max_length: int=135) -> List[str]:
    """
    Chunk text into smaller pieces.
    """
    chunks = []

    current_chunk = ""
    current_chunk_length = 0

    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        encoded = sentence.encode("utf-8")
        encoded_length = len(encoded)

        if encoded_length == 0:
            continue

        if current_chunk_length + encoded_length <= max_length:
            current_chunk += sentence
            current_chunk_length += encoded_length
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_chunk_length = encoded_length

        if not ends_with_multi_byte_character(sentence):
            current_chunk += " "
            current_chunk_length += 1

    if current_chunk_length > 0:
        chunks.append(current_chunk.strip())

    return chunks
