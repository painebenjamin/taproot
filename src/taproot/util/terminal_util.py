from typing import Any, Iterable, Iterator, Optional

__all__ = [
    "green",
    "red",
    "yellow",
    "blue",
    "cyan",
    "magenta",
    "maybe_use_tqdm",
    "maybe_use_termcolor",
]

def tqdm_is_available() -> bool:
    """
    Return whether tqdm is available.

    :return: Whether tqdm is available.
    """
    try:
        import tqdm
        return True
    except ImportError:
        return False

def termcolor_is_available() -> bool:
    """
    Return whether termcolor is available.

    :return: Whether termcolor is available.
    """
    try:
        import termcolor
        return True
    except ImportError:
        return False

def maybe_use_tqdm(
    iterable: Iterable[Any],
    use_tqdm: bool=True,
    desc: Optional[str]=None,
    total: Optional[int]=None,
    unit: str="it",
    unit_scale: bool=False,
    unit_divisor: int=1000
) -> Iterator[Any]:
    """
    Return the iterable or wrap it in a tqdm if use_tqdm is True.

    :param iterable: The iterable to return.
    :param use_tqdm: Whether to wrap the iterable in a tqdm.
    :param desc: The description to display.
    :param total: The total number of items.
    :param unit: The unit to display.
    :param unit_scale: Whether to scale the unit.
    :param unit_divisor: The unit divisor.
    :return: The iterable or tqdm wrapped iterable.
    """
    if use_tqdm and tqdm_is_available():
        from tqdm import tqdm
        for item in tqdm(iterable, desc=desc, total=total, unit=unit, unit_scale=unit_scale, unit_divisor=unit_divisor):
            yield item
    else:
        for item in iterable:
            yield item

def maybe_use_termcolor(
    message: str,
    color: Optional[str]=None,
    **kwargs: Any
) -> str:
    """
    Return the message with color if termcolor is available.

    :param message: The message to display.
    :param color: The color to use.
    :param kwargs: Additional keyword arguments.
    :return: The formatted message.
    """
    if color is not None and termcolor_is_available():
        import termcolor
        return termcolor.colored(message, color, **kwargs) # type: ignore[arg-type]
    return message

def green(
    message: str,
    light: bool=True
) -> str:
    """
    Return a green message.

    :param message: The message to display.
    :param light: Whether to use a light green color.
    :return: The formatted message.
    """
    return maybe_use_termcolor(message, "light_green" if light else "green")

def red(
    message: str,
    light: bool=True
) -> str:
    """
    Return an red message.

    :param message: The message to display.
    :param light: Whether to use a light red color.
    :return: The formatted message.
    """
    return maybe_use_termcolor(message, "light_red" if light else "red")

def yellow(
    message: str,
    light: bool=True
) -> str:
    """
    Return a yellow message.

    :param message: The message to display.
    :param light: Whether to use a light yellow color.
    :return: The formatted message.
    """
    return maybe_use_termcolor(message, "light_yellow" if light else "yellow")

def blue(
    message: str,
    light: bool=True
) -> str:
    """
    Return an blue message.

    :param message: The message to display.
    :param light: Whether to use a light blue color.
    :return: The formatted message.
    """
    return maybe_use_termcolor(message, "light_blue" if light else "blue")

def cyan(
    message: str,
    light: bool=True
) -> str:
    """
    Return a cyan message.

    :param message: The message to display.
    :param light: Whether to use a light cyan color.
    :return: The formatted message.
    """
    return maybe_use_termcolor(message, "light_cyan" if light else "cyan")

def magenta(
    message: str,
    light: bool=True
) -> str:
    """
    Return a magenta message.

    :param message: The message to display.
    :param light: Whether to use a light magenta color.
    :return: The formatted message.
    """
    return maybe_use_termcolor(message, "light_magenta" if light else "magenta")
