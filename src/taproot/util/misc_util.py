from __future__ import annotations

import os
import sys
import asyncio
import nest_asyncio # type: ignore[import-untyped]

from typing import Any, Iterable, List, Optional, Union, Callable, Coroutine, Dict, Tuple, Iterator, TYPE_CHECKING

from math import ceil, log10
from itertools import islice
from functools import lru_cache, wraps
from time import monotonic_ns, perf_counter
from uuid import uuid4

if TYPE_CHECKING:
    from ..payload import (
        ParameterMetadataPayload,
        TaskPayload,
        FlexibleResultMapping
    )

__all__ = [
    "reiterator",
    "generate_id",
    "get_payload_id",
    "get_parameter_metadata",
    "get_metadata",
    "estimate_parameter_bytes",
    "chunk_bytes",
    "chunk_iterable",
    "timed_lru_cache",
    "asyncio_run",
    "get_step_callback",
    "get_step_iterator",
    "sliding_windows",
    "sliding_window_count",
    "no_op",
    "merge_into",
    "floor_power",
    "get_continuation_depth",
    "get_parameters_from_result",
    "async_input",
    "package_is_available",
    "get_secret"
]

class reiterator:
    """
    Transparently memoized any iterator
    """
    memoized: List[Any]

    def __init__(self, iterable: Iterable[Any]) -> None:
        self.iterable = iterable
        self.memoized = []
        self.started = False
        self.finished = False

    def __iter__(self) -> Iterable[Any]:
        if not self.started:
            self.started = True
            last_index: Optional[int] = None
            for i, value in enumerate(self.iterable):
                yield value
                self.memoized.append(value)
                last_index = i
                if self.finished:
                    # Completed somewhere else
                    break
            if self.finished:
                if last_index is None:
                    last_index = 0
                for value in self.memoized[last_index+1:]:
                    yield value
            self.finished = True
            del self.iterable
        elif not self.finished:
            # Complete iterator
            self.memoized += [item for item in self.iterable]
            self.finished = True
            del self.iterable
            for item in self.memoized:
                yield item
        else:
            for item in self.memoized:
                yield item

def generate_id() -> str:
    """
    Generate a unique identifier.

    :return: A unique identifier (UUIDv4).
    """
    return str(uuid4())

def get_payload_id(payload: Dict[str, Any]) -> str:
    """
    Get a unique identifier for a payload.

    :param payload: The payload to get the identifier for.
    :return: The unique identifier for the payload.
    """
    from hashlib import sha256
    from .introspection_util import is_torch_tensor, is_pil_image, is_numpy_array
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Payload must be a non-empty dictionary.")

    def _dict_to_sorted_tuple(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        """
        Convert a dictionary to a sorted tuple.
        """
        return tuple(sorted((k, _dict_to_sorted_tuple(v) if isinstance(v, dict) else v) for k, v in d.items()))

    hasher = sha256()
    def _update_hash(tuples: Tuple[Tuple[str, Any], ...]) -> None:
        for key, value in tuples:
            hasher.update(key.encode())
            if isinstance(value, tuple):
                _update_hash(value)
            else:
                if is_torch_tensor(value):
                    value = value.detach().cpu().numpy().tobytes()
                elif is_pil_image(value):
                    value = value.tobytes()
                elif is_numpy_array(value):
                    value = value.tobytes()
                else:
                    value = str(value).encode()
                hasher.update(value)
    _update_hash(_dict_to_sorted_tuple(payload))
    return hasher.hexdigest()

def get_metadata(parameters: Dict[str, Any]) -> Dict[str, ParameterMetadataPayload]:
    """
    Get the metadata for a set of parameters.

    :param parameters: The parameters to get the metadata for.
    :return: The metadata for the parameters.
    """
    metadata: Dict[str, ParameterMetadataPayload] = {}
    for key, value in parameters.items():
        metadata[key] = get_parameter_metadata(value)
    return metadata

def get_parameter_metadata(parameter: Any) -> ParameterMetadataPayload:
    """
    Gets the metadata for a parameter.
    Doesn't actually import the types to avoid circular imports and unnecessary dependencies.

    :param parameter: The parameter to get the metadata for.
    :return: The metadata for the parameter.
    """
    from .log_util import logger

    parameter_type = type(parameter)
    parameter_type_names = [
        getattr(mro_type, "__name__", str(mro_type))
        for mro_type in parameter_type.mro()
    ]
    metadata: ParameterMetadataPayload = {
        "parameter_type": parameter_type,
        "parameter_sub_metadata": None,
        "parameter_size": None
    }

    if isinstance(parameter, list):
        parameter_size = len(parameter)
        metadata["parameter_size"] = (parameter_size,)
        if parameter_size > 0:
            metadata["parameter_sub_metadata"] = get_parameter_metadata(parameter[0])
    elif isinstance(parameter, tuple):
        metadata["parameter_sub_metadata"] = tuple([get_parameter_metadata(value) for value in parameter])
        metadata["parameter_size"] = (len(parameter),)
    elif isinstance(parameter, dict):
        metadata["parameter_sub_metadata"] = {}
        for key, value in parameter.items():
            metadata["parameter_sub_metadata"][key] = get_parameter_metadata(value) # type: ignore[index,literal-required]
        metadata["parameter_size"] = (len(parameter),)
    elif isinstance(parameter, str):
        if os.path.exists(parameter):
            metadata["parameter_size"] = (os.path.getsize(parameter),)
        else:
            metadata["parameter_size"] = (len(parameter),)
    elif "Image" in parameter_type_names:
        try:
            metadata["parameter_size"] = parameter.size
        except:
            logger.debug(f"Thought {parameter} was an image, but it doesn't have a size attribute.")
            pass
    elif "Tensor" in parameter_type_names:
        try:
            metadata["parameter_size"] = parameter.size()
        except:
            logger.debug(f"Thought {parameter} was a tensor, but it doesn't have a size method.")
            pass
    elif "ndarray" in parameter_type_names:
        try:
            metadata["parameter_size"] = parameter.shape
        except:
            logger.debug(f"Thought {parameter} was a numpy array, but it doesn't have a shape attribute.")
            pass
    elif hasattr(parameter, "__len__"):
        metadata["parameter_size"] = (len(parameter),)

    return metadata

def estimate_parameter_bytes(
    *parameter_args: ParameterMetadataPayload,
    **parameter_kwargs: ParameterMetadataPayload
) -> int:
    """
    Given a payload of metadata, estimate the number of bytes required to store the parameters.
    
    :param parameters: The parameters to estimate the size of.
    :return: The estimated number of bytes required to store the parameters.
    """
    total_parameter_bytes: int = 0

    def add_parameter_bytes(metadata: ParameterMetadataPayload, multiplier: int=1) -> None:
        nonlocal total_parameter_bytes
        parameter_size = metadata.get("parameter_size", None)
        if parameter_size is not None:
            if isinstance(parameter_size, tuple) or isinstance(parameter_size, list):
                # The size is the product of the numbers in the tuple
                total_parameter_size = parameter_size[0]
                for size in parameter_size[1:]:
                    total_parameter_size *= size
                parameter_size = total_parameter_size // 2 # type: ignore[assignment]
            total_parameter_bytes += parameter_size # type: ignore[operator]
        sub_metadata = metadata.get("parameter_sub_metadata", {})
        if sub_metadata and isinstance(sub_metadata, dict):
            sub_metadata_multiplier = 1 if parameter_size is None else parameter_size
            if "parameter_type" in sub_metadata:
                add_parameter_bytes(sub_metadata, sub_metadata_multiplier) # type: ignore[arg-type]
            else:
                for sub_metadata_dict in sub_metadata.values():
                    if isinstance(sub_metadata_dict, dict):
                        add_parameter_bytes(sub_metadata_dict, sub_metadata_multiplier) # type: ignore[arg-type]

    for parameter_metadata in parameter_args:
        add_parameter_bytes(parameter_metadata)

    for parameter_metadata in parameter_kwargs.values():
        add_parameter_bytes(parameter_metadata)

    return total_parameter_bytes

def chunk_iterable(
    iterable: Iterable[Any],
    chunk_size: int,
    pad_to_size: bool = False,
    pad_with: Any = None,
) -> Iterable[List[Any]]:
    """
    Split an iterable into chunks of a given size.

    >>> list(chunk_iterable(range(10), 3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    >>> list(chunk_iterable(range(10), 3, pad_to_size=True))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]
    >>> list(chunk_iterable(range(10), 3, pad_to_size=True, pad_with='x'))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 'x', 'x']]

    :param iterable: The iterable to split.
    :param chunk_size: The size of the chunks.
    :param pad_to_size: Whether to pad the last chunk to the chunk size.
    :param pad_with: The value to pad with.
    :return: An iterator over the chunks.
    """
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        if pad_to_size and len(chunk) < chunk_size:
            chunk += [pad_with] * (chunk_size - len(chunk))
        yield chunk

def chunk_bytes(
    data: bytes,
    chunk_size: int,
    pad_to_size: bool = False,
    pad_with: bytes = b"\x00",
) -> Iterable[bytes]:
    """
    Split a bytes object into chunks of a given size.

    >>> list(chunk_bytes(b"abcdefghij", 3))
    [b'abc', b'def', b'ghi', b'j']
    >>> list(chunk_bytes(b"abcdefghij", 3, pad_to_size=True))
    [b'abc', b'def', b'ghi', b'j\x00\x00']

    :param data: The bytes object to split.
    :param chunk_size: The size of the chunks.
    :param pad_to_size: Whether to pad the last chunk to the chunk size.
    :param pad_with: The byte to pad with.
    :return: An iterator over the chunks.
    """
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        if pad_to_size and len(chunk) < chunk_size:
            chunk += pad_with * (chunk_size - len(chunk))
        yield chunk

def timed_lru_cache(
    _func: Optional[Callable[[Any], Any]]=None,
    *,
    ttl: Union[int,float] = 7000,
    maxsize: int = 128,
    typed: bool = False
) -> Callable[[Any], Any]:
    """
    Extension over existing lru_cache with timeout
    Thanks to https://blog.soumendrak.com/cache-heavy-computation-functions-with-a-timeout-value

    :param _func: The function to cache.
    :param ttl: The time to live for the cache in seconds.
    :param maxsize: The maximum size of the cache.
    :param typed: Whether to differentiate between arguments of different types.
    :return: A decorator for the function.
    """

    def wrapper_cache(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
        # create a function wrapped with traditional lru_cache
        f = lru_cache(maxsize=maxsize, typed=typed)(f)
        # convert seconds to nanoseconds to set the expiry time in nanoseconds
        f.delta = ttl * 10 ** 9 # type: ignore[attr-defined]
        f.expiration = monotonic_ns() + f.delta # type: ignore[attr-defined]

        @wraps(f)  # wraps is used to access the decorated function attributes
        def wrapped_f(*args: Any, **kwargs: Any) -> Any:
            if monotonic_ns() >= f.expiration: # type: ignore[attr-defined]
                # if the current cache expired of the decorated function then 
                # clear cache for that function and set a new cache value with new expiration time 
                f.cache_clear()
                f.expiration = monotonic_ns() + f.delta # type: ignore[attr-defined]
            return f(*args, **kwargs)

        wrapped_f.cache_info = f.cache_info # type: ignore[attr-defined]
        wrapped_f.cache_clear = f.cache_clear # type: ignore[attr-defined]
        return wrapped_f

    # To allow decorator to be used without arguments
    if _func is None:
        return wrapper_cache
    else:
        return wrapper_cache(_func)

def asyncio_run(
    future: Union[asyncio.Future[Any], asyncio.Task[Any], Coroutine[Any, Any, Any]],
    *,
    as_task: bool=True,
) -> Any:
    """
    A better implementation of `asyncio.run`.

    :param future: A future or task or call of an async method.
    :param as_task: Forces the future to be scheduled as task (needed for e.g. aiohttp).
    :return: The result of the future.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no event loop running:
        loop = asyncio.new_event_loop()
        if not as_task or isinstance(future, asyncio.Task):
            task = future
        else:
            task = loop.create_task(future) # type: ignore[arg-type]
        return loop.run_until_complete(task)
    else:
        nest_asyncio.apply(loop)
        if not as_task or isinstance(future, asyncio.Task):
            task = future
        else:
            task = loop.create_task(future) # type: ignore[arg-type]
        return asyncio.run(task) # type: ignore[arg-type]

def get_step_callback(
    overall_steps: int,
    task: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    log_interval: int = 5,
    log_sampling_duration: Union[int, float] = 2,
    log_durations: bool=True,
    alpha: float = 0.1 # EMA smoothing
) -> Callable[..., None]:
    """
    Creates a scoped callback to trigger during iterations

    :param overall_steps: The total number of steps.
    :param task: The task to log.
    :param progress_callback: The callback to call with progress.
    :param log_interval: The interval to log at.
    :param log_sampling_duration: The duration to sample for logging.
    :param log_durations: Whether to log durations.
    :param alpha: The EMA smoothing factor.
    :return: A callback to call when a step is completed.
    """
    from .log_util import logger
    from .string_util import human_duration

    digits = ceil(log10(overall_steps))
    log_prefix = "" if not task else f"[{task}] "
    overall_step, ema_rate = 0, 0.0
    start_time = perf_counter()

    def step_complete(increment_step: bool = True) -> None:
        """
        Called when a step is completed.
        """
        nonlocal overall_step, ema_rate, start_time
        now = perf_counter()
        if increment_step:
            overall_step += 1

        # Calculate rate
        instantaneous_rate = 0.0
        if overall_step != 0:
            instantaneous_rate = overall_step / (now - start_time)

        # Update EMA rate
        if overall_step != 1:
            ema_rate = alpha * instantaneous_rate + (1 - alpha) * ema_rate
        else:
            ema_rate = instantaneous_rate

        if overall_step != 0 and overall_step % log_interval == 0 or overall_step == overall_steps:
            unit = "s/it" if ema_rate < 1 else "it/s"
            its_display = 0 if ema_rate == 0 else 1 / ema_rate if ema_rate < 1 else ema_rate
            durations = ""
            if log_durations:
                seconds_elapsed = now - start_time
                seconds_remaining = (overall_steps - overall_step) / ema_rate
                durations = " [" + human_duration(seconds_elapsed)
                durations += "<" + human_duration(seconds_remaining) + "]"

            logger.debug(f"{log_prefix}{overall_step:0{digits}d}/{overall_steps:0{digits}d}: {ema_rate:0.2f} {unit}{durations}")

        if progress_callback is not None:
            progress_callback(overall_step, overall_steps, ema_rate)

    return step_complete

def get_step_iterator(
    items: Iterable[Any],
    task: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    log_interval: int = 5,
    log_sampling_duration: Union[int, float] = 2,
) -> Iterator[Any]:
    """
    Gets an iterator over items that will call the progress function over each item.

    :param items: The items to iterate over.
    :param task: The task to log.
    :param progress_callback: The callback to call with progress.
    :param log_interval: The interval to log at.
    :param log_sampling_duration: The duration to sample for logging.
    :return: An iterator over the items.
    """
    items = [item for item in items]
    callback = get_step_callback(
        len(items),
        task=task,
        progress_callback=progress_callback,
        log_interval=log_interval,
        log_sampling_duration=log_sampling_duration
    )
    for item in items:
        yield item
        callback(True)

def sliding_windows(
    height: int,
    width: int,
    tile_size: Union[int, Tuple[int, int]],
    tile_stride: Union[int, Tuple[int, int]],
) -> List[Tuple[int, int, int, int]]:
    """
    Gets windows over a height/width using a square tile.

    :param height: The height of the area.
    :param width: The width of the area.
    :param tile_size: The size of the tile.
    :param tile_stride: The stride of the tile.
    """
    if isinstance(tile_size, tuple):
        tile_width, tile_height = tile_size
    else:
        tile_width = tile_height = tile_size

    if isinstance(tile_stride, tuple):
        tile_stride_width, tile_stride_height = tile_stride
    else:
        tile_stride_width = tile_stride_height = tile_stride

    height_list = list(range(0, height - tile_height + 1, tile_stride_height))
    if (height - tile_height) % tile_stride_height != 0:
        height_list.append(height - tile_height)

    width_list = list(range(0, width - tile_width + 1, tile_stride_width))
    if (width - tile_width) % tile_stride_width != 0:
        width_list.append(width - tile_width)

    coords: List[Tuple[int, int, int, int]] = []
    for height in height_list:
        for width in width_list:
            coords.append((height, height + tile_height, width, width + tile_width))

    return coords

def sliding_window_count(
	height: int,
    width: int,
    tile_size: Union[int, Tuple[int, int]],
    tile_stride: Union[int, Tuple[int, int]],
) -> int:
    """
    Calculate the number of tiles needed to cover a rectangular area.

    :param height: The height of the area.
    :param width: The width of the area.
    :param tile_size: The size of the tile.
    :param tile_stride: The stride of the tile.
    :return: The number of tiles needed to cover the area.
    """
    from math import ceil
    if isinstance(tile_size, tuple):
        tile_width, tile_height = tile_size
    else:
        tile_width = tile_height = tile_size

    if isinstance(tile_stride, tuple):
        tile_stride_width, tile_stride_height = tile_stride
    else:
        tile_stride_width = tile_stride_height = tile_stride

    return (
        ceil((height - tile_height) / tile_stride_height + 1) *
        ceil((width - tile_width) / tile_stride_width + 1)
    )

def no_op(*args: Any, **kwargs: Any) -> None:
    """
    Does nothing.
    """

def merge_into(source: Dict[str, Any], dest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges a source dictionary into a target dictionary.

    >>> x = {"a": 1}
    >>> r = merge_into({"b": 2}, x)
    >>> r
    {'a': 1, 'b': 2}
    >>> r is x
    True

    :param source: The source dictionary to merge.
    :param dest: The destination dictionary to merge into.
    :return: The destination dictionary.
    """
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(dest.get(key, None), dict):
            merge_into(source[key], dest[key])
        else:
            dest[key] = source[key]
    return dest

def floor_power(number: int, power: int=2) -> int:
    """
    Get the floor power of a number.
    
    >>> floor_power(10, 2)
    8
    >>> floor_power(10, 3)
    9
    >>> floor_power(10, 4)
    4
    >>> floor_power(10, 11)
    1

    :param number: The number to get the floor power of.
    :param power: The power to use.
    """
    floor = 1
    while floor <= number:
        next_floor = floor * power
        if next_floor > number:
            break
        floor = next_floor
    return floor

def get_continuation_depth(payload: TaskPayload, limit: int=100) -> int:
    """
    Get the depth of a continuation from a task payload.
    Ensures recursions do not exceed the maximum recursion depth, which
    is much smaller than python's recursion limit.

    >>> get_continuation_depth({})
    0
    >>> get_continuation_depth({"continuation": {}})
    1
    >>> get_continuation_depth({"continuation": {"continuation": {}}})
    2

    :param payload: The payload to get the continuation depth from.
    :param limit: The limit to use for recursion.
    :return: The depth of the continuation.
    """
    current_recursion_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(limit * 10)
        if "continuation" in payload:
            payload_continuation = payload["continuation"]
            if isinstance(payload_continuation, list):
                return 1 + max(get_continuation_depth(item) for item in payload_continuation)
            else:
                return 1 + get_continuation_depth(payload_continuation)
        return 0
    except RecursionError:
        return limit + 1
    finally:
        sys.setrecursionlimit(current_recursion_limit)

def get_parameters_from_result(
    result: Any,
    result_map: FlexibleResultMapping
) -> Any:
    """
    Get parameters from a map of parameter names to values.

    :param result: The result to map.
    :param result_map: The map of parameter names to values.
    :return: The parameters from the result.
    """
    if isinstance(result_map, str):
        return {result_map: result}
    elif isinstance(result_map, dict):
        assert isinstance(result, dict), "Result must be a dictionary when using a dictionary result map."
        parameters = {}
        for key, value in result_map.items():
            parameters[key] = get_parameters_from_result(result[key], value)
        return parameters
    elif isinstance(result_map, list):
        assert isinstance(result, list), "Result must be a list when using a list result map."
        assert len(result) == len(result_map), "Result and result map must have the same length."
        return [
            get_parameters_from_result(item, map_item)
            for item, map_item in zip(result, result_map)
        ]
    else:
        return result

async def async_input(message: str = "") -> str:
    """
    Gets input from stdin in an async-friendly manner.

    :param message: The message to display before getting input.
    :return: The input from stdin.
    """
    write = lambda: sys.stdout.write(message + " ")
    await asyncio.get_event_loop().run_in_executor(None, write)
    return await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)

def package_is_available(package_name: str) -> bool:
    """
    Checks if a package is available.

    :param package_name: The name of the package to check.
    :return: True if the package is available, False otherwise.
    """
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def get_secret(secret_key: str) -> Optional[str]:
    """
    Gets a secret from the environment, with multiple possible keys.

    :param secret_key: The key to get the secret for.
    :return: The secret if it exists, None otherwise.
    """
    secret_key = secret_key.upper()
    return os.getenv(
        secret_key,
        os.getenv(f"SECRET_{secret_key}", None)
    )
