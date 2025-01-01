from __future__ import annotations

import os
import sys
import asyncio
import weakref
import warnings
import tracemalloc

from time import perf_counter
from types import TracebackType
from typing import Set, List, Optional, Iterator, Any, Type, Dict, Tuple, Union, TYPE_CHECKING
from typing_extensions import Literal
from random import choice, choices
from contextlib import contextmanager

from .audio_util import EncodedAudioProxy
from .number_util import NumericMixin

if TYPE_CHECKING:
    import torch
    from PIL import Image
    from ..tasks import Task

__all__ = [
    "profiler",
    "log_duration",
    "restrict_module_import",
    "get_test_restricted_import_context",
    "get_test_server_addresses",
    "get_test_server_protocols",
    "RAMCounter",
    "VRAMCounter",
    "TimeCounter",
    "ram_counter",
    "vram_counter",
    "time_counter",
    "assert_exception",
    "restrict_gpu",
    "get_test_audio",
    "get_test_images",
    "get_test_image",
    "get_test_results",
    "get_test_result",
    "save_test_image",
    "save_test_audio",
    "save_test_video",
    "get_image_similarity",
    "get_text_similarity",
    "assert_image_equality",
    "assert_text_equality",
    "assert_test_output",
    "execute_task_test_suite",
]

@contextmanager
def log_duration(name: Optional[str]=None) -> Iterator[None]:
    """
    A context manager that prints the time taken to enter and exit the context.
    """
    from .log_util import logger
    to_raise: Optional[Exception] = None
    with time_counter() as counter:
        try:
            yield
        except Exception as ex:
            to_raise = ex
            pass
    method_name = "method" if not name else name
    if to_raise is None:
        logger.info(f"{method_name} took {counter}")
    else:
        logger.error(f"{method_name} took {counter} to produce an exception")
        raise to_raise

@contextmanager
def profiler() -> Iterator[None]:
    """
    Runs a profiler.
    """
    from cProfile import Profile
    from pstats import SortKey, Stats
    from .log_util import logger
    with Profile() as profile:
        try:
            yield
        finally:
            Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats()

@contextmanager
def restrict_module_import(*modules: str) -> Iterator[None]:
    """
    A context manager that restricts the import of certain modules.
    """
    import sys
    original_import = sys.modules["builtins"].__import__
    modules = tuple(module.lower() for module in modules)

    from .log_util import logger
    logger.debug(f"Restricting import of {modules}")
    def restricted_import(
        name: str,
        locals: Optional[Dict[str, Any]]=None,
        globals: Optional[Dict[str, Any]]=None,
        fromlist: Optional[List[str]]=None,
        level: int=0
    ) -> Any:
        """
        A restricted import function that raises an ImportError if the module is restricted.
        Otherwise, it calls the original import function.
        """
        base_name = name.split(".")[0].lower()
        if base_name in modules and level == 0:
            raise ImportError(
                f"Import of {base_name} is restricted. " +
                "If you are testing a task, be sure to set " +
                "`required_packages()` include necessary modules " +
                f"to lift access restrictions."
            )
        return original_import(name, locals, globals, fromlist, level)

    sys.modules["builtins"].__import__ = restricted_import # type: ignore[attr-defined]
    try:
        yield
    finally:
        sys.modules["builtins"].__import__ = original_import # type: ignore[attr-defined]

@contextmanager
def get_test_restricted_import_context(*allowed_modules: str) -> Iterator[None]:
    """
    A context manager that restricts the import of certain modules.
    Uses the superset of restricted modules from the constants file, then
    allows certain modules to be imported by removing them from the restricted list.
    In practice, this should be used to enforce tasks to define their required modules.
    """
    from ..constants import TEST_RESTRICTED_MODULES
    SCIPY_MODULES = {"skimage", "sklearn", "scikit-image", "scikit-learn", "diffusers"}
    TORCH_MODULES = {"torchvision", "torchaudio", "pytorch_lightning", "diffusers", "transformers", "peft"}

    allowed_module_set = set([allowed_module.lower() for allowed_module in allowed_modules])

    if any(module in allowed_module_set for module in SCIPY_MODULES):
        allowed_module_set.add("scipy")
    if any(module in allowed_module_set for module in TORCH_MODULES):
        allowed_module_set.add("torch")
    with restrict_module_import(*(TEST_RESTRICTED_MODULES - allowed_module_set)):
        yield

def get_test_server_protocols(no_memory: bool=False) -> List[str]:
    """
    Returns a list of test server configurations, in the form of (address, is_secure).
    Addresses are parsed according to the specifications set out in `network_util.parse_address`.
    """
    protocols = []
    if not no_memory:
        protocols.append("memory")
    protocols.extend(["tcp", "ws"])
    if sys.platform != "win32":
        protocols.append("unix")
    protocols.reverse()
    return protocols

def get_test_server_addresses(no_memory: bool=False) -> List[str]:
    """
    Returns a list of test server configurations, in the form of (address, is_secure).
    Addresses are parsed according to the specifications set out in `network_util.parse_address`.
    """
    from .network_util import (
        find_free_port,
        find_free_memory_port,
        find_free_unix_socket,
    )

    configurations = []
    if not no_memory:
        configurations.append(f"memory://{find_free_memory_port()}")
    configurations.extend([
        f"tcp://127.0.0.1:{find_free_port()}",
        f"tcps://127.0.0.1:{find_free_port()}",
        f"ws://127.0.0.1:{find_free_port()}",
        f"wss://127.0.0.1:{find_free_port()}"
    ])
    if sys.platform != "win32":
        configurations.append(f"unix://{find_free_unix_socket()}")
    configurations.reverse()
    return configurations

class RAMCounter(NumericMixin):
    """
    A context manager that tracks the maximum memory usage of a block of code.

    >>> with RAMCounter() as counter: \
            large_list = [i for i in range(100000)]
    >>> counter.max_memory_usage
    3595620
    >>> str(counter)
    '3.60 MB'
    """
    _start_snapshot: tracemalloc.Snapshot
    _end_snapshot: tracemalloc.Snapshot
    _max_memory_usage: int

    def __enter__(self) -> RAMCounter:
        """
        Starts tracking memory usage.
        """
        tracemalloc.start()
        self._start_snapshot = tracemalloc.take_snapshot()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType]
    ) -> None:
        """
        Stops tracking memory usage.
        """
        self._end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

    @property
    def max_memory_usage(self) -> int:
        """
        Returns the maximum memory usage during the block of code.
        """
        if not hasattr(self, "_max_memory_usage"):
            if not hasattr(self, "_start_snapshot") or not hasattr(self, "_end_snapshot"):
                raise RuntimeError("RAMCounter is not active, you must use it (using the `with` keyword) before you access the memory usage.")

            start_stats = self._start_snapshot.statistics('lineno')
            end_stats = self._end_snapshot.statistics('lineno')

            start_memory = sum(stat.size for stat in start_stats)
            end_memory = sum(stat.size for stat in end_stats)

            self._max_memory_usage = end_memory - start_memory
        return self._max_memory_usage

    @property
    def numeric(self) -> int:
        """
        Returns the maximum memory usage during the block of code.
        """
        return self.max_memory_usage

    def __str__(self) -> str:
        """
        Returns the maximum memory usage in human-readable form.
        """
        from .string_util import human_size
        return human_size(self.max_memory_usage)

def ram_counter() -> RAMCounter:
    """
    A function that returns a RAMCounter instance.
    """
    return RAMCounter()

class VRAMCounter(NumericMixin):
    """
    A context manager that tracks the maximum VRAM usage of a block of code using PyTorch.

    >>> torch = __import__("torch")
    >>> with VRAMCounter() as counter: \
            large_tensor = torch.randn(1000, 1000).to("cuda")
    >>> counter.max_memory_usage
    4000256
    >>> str(counter)
    '4.00 MB'
    """
    _start_memory: int
    _peak_memory: int
    _max_memory_usage: int
    _task: asyncio.Task[Any]

    def __init__(
        self,
        device: Optional[torch.device]=None,
        polling_interval: float=0.01
    ) -> None:
        import torch
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            raise RuntimeError("No device available for memory tracking.")
        self.polling_interval = polling_interval

    def _get_allocated_memory(self) -> int:
        """
        Gets the allocated memory on the device.
        """
        import torch
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device)
        elif self.device.type == 'mps':
            return torch.mps.current_allocated_memory()
        else:
            raise RuntimeError("No known method to track memory usage for device type '{self.device.type}'")

    def _update_peak_memory(self) -> None:
        """
        Updates the peak memory usage.
        """
        self._peak_memory = max(self._peak_memory, self._get_allocated_memory())

    def _check_stop_task(self, task: asyncio.Task[Any]) -> None:
        """
        Checks if the task is still running and stops it if it is.
        """
        if hasattr(self, "_task") and not task.done():
            task.cancel()

    async def _poll_memory(self) -> None:
        """
        Polls the memory usage of the device.
        """
        while True:
            self._update_peak_memory()
            try:
                await asyncio.sleep(self.polling_interval)
            except asyncio.CancelledError:
                break

    def __enter__(self) -> VRAMCounter:
        """
        Starts tracking memory usage.
        """
        self._start_memory = self._get_allocated_memory()
        self._peak_memory = self._start_memory
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self._task = loop.create_task(self._poll_memory())
        weakref.finalize(self, self._check_stop_task, self._task)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType]
    ) -> None:
        """
        Stops tracking memory usage.
        """
        self._task.cancel()
        self._update_peak_memory()

    @property
    def max_memory_usage(self) -> int:
        """
        Returns the maximum memory usage during the block of code.
        """
        if not hasattr(self, "_max_memory_usage"):
            if not hasattr(self, "_start_memory") or not hasattr(self, "_peak_memory"):
                raise RuntimeError("VRAMCounter is not active, you must use it (using the `with` keyword) before you access the memory usage.")
            self._max_memory_usage = self._peak_memory - self._start_memory
        return self._max_memory_usage

    @property
    def numeric(self) -> int:
        """
        Returns the maximum memory usage during the block of code.
        """
        return self.max_memory_usage

    def __str__(self) -> str:
        """
        Returns the maximum memory usage in human-readable form.
        """
        from .string_util import human_size
        return human_size(self.max_memory_usage)

class NvidiaSMIVRAMCounter(VRAMCounter):
    """
    A class which uses the external nvidia-smi command to track the maximum VRAM usage of a block of code.

    This is useful for when the PyTorch memory tracking is not accurate or it is running in a separate process, like llama.cpp.
    """
    def _get_allocated_memory(self) -> int:
        """
        Returns the allocated memory on the device.
        """
        import subprocess
        command = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError("nvidia-smi command failed.")
        result_text = result.stdout.decode().strip()
        return int(result_text) * 1000 * 1000

def vram_counter(
    device: Optional[torch.device]=None,
    use_nvidia_smi: bool=False,
) -> VRAMCounter:
    """
    A function that returns a VRAMCounter instance.
    """
    if use_nvidia_smi:
        return NvidiaSMIVRAMCounter(device=device)
    return VRAMCounter(device=device)

class TimeCounter(NumericMixin):
    """
    A context manager that tracks the time taken to execute a block of code.
    """
    _start_time: float
    _end_time: float
    _elapsed_time: float

    def __enter__(self) -> TimeCounter:
        """
        Starts tracking time.
        """
        self._start_time = perf_counter()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[Type[BaseException]]) -> None:
        """
        Stops tracking time.
        """
        self._end_time = perf_counter()

    @property
    def elapsed_time(self) -> float:
        """
        Returns the time taken to execute the block of code.
        """
        if not hasattr(self, "_elapsed_time"):
            if not hasattr(self, "_start_time") or not hasattr(self, "_end_time"):
                raise RuntimeError("TimeCounter is not active, you must use it (using the `with` keyword) before you access the elapsed time.")
            self._elapsed_time = self._end_time - self._start_time
        return self._elapsed_time

    @property
    def numeric(self) -> float:
        """
        Returns the time taken to execute the block of code.
        """
        return self.elapsed_time

    def __str__(self) -> str:
        """
        Returns the time taken to execute the block of code in human-readable form.
        """
        from .string_util import human_duration
        return human_duration(self.elapsed_time)

def time_counter() -> TimeCounter:
    """
    A function that returns a TimeCounter instance.
    """
    return TimeCounter()

def assert_exception(
    exception: Type[BaseException],
    func: Any,
    *args: Any,
    **kwargs: Any
) -> None:
    """
    Asserts that a function raises an exception with a specific message.

    >>> def raise_exception() -> None: \
            raise ValueError("This is a test exception")
    >>> assert_exception(ValueError, raise_exception)
    """
    try:
        func(*args, **kwargs)
    except Exception as e:
        assert isinstance(e, exception)
    else:
        raise AssertionError(f"Expected {exception} to be raised.")

@contextmanager
def restrict_gpu() -> Iterator[None]:
    """
    A context manager that restricts the use of CUDA.

    >>> import torch
    >>> with restrict_gpu(): \
            assert_exception(RuntimeError, lambda: torch.randn(1000, 1000).to("cuda"))
    """
    import torch
    def raise_exception(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "GPU access is restricted. " +
            "If you are testing a task, be sure to set " +
            "`requires_gpu` to True in order to lift access restrictions."
        )
    # These methods are directly intervened.
    original_cuda_is_available = torch.cuda.is_available
    original_init = torch.cuda.init
    original_is_initialized = torch.cuda.is_initialized
    original_device_count = torch.cuda.device_count
    original_current_device = torch.cuda.current_device
    original_get_device_name = torch.cuda.get_device_name
    original_get_device_capability = torch.cuda.get_device_capability
    original_get_device_properties = torch.cuda.get_device_properties
    original_cuda = torch.Tensor.cuda
    torch.cuda.is_available = raise_exception # type: ignore[assignment]
    torch.cuda.init = raise_exception
    torch.cuda.is_initialized = raise_exception
    torch.cuda.device_count = raise_exception # type: ignore[assignment]
    torch.cuda.current_device = raise_exception # type: ignore[assignment]
    torch.cuda.get_device_name = raise_exception # type: ignore[assignment]
    torch.cuda.get_device_capability = raise_exception # type: ignore[assignment]
    torch.cuda.get_device_properties = raise_exception # type: ignore[assignment]
    torch.Tensor.cuda = raise_exception # type: ignore[assignment,method-assign]

    # We allow to() to mass through to cpu if it is called.
    original_to = torch.Tensor.to
    def maybe_raise_exception( # type: ignore[return]
        self: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        device = args[0] if args else kwargs.get("device", None)
        if device == "cpu" or isinstance(device, torch.device) and device.type == "cpu":
            return original_to(self, *args, **kwargs)
        raise_exception()

    torch.Tensor.to = maybe_raise_exception # type: ignore[assignment,method-assign]

    try:
        yield
    finally:
        torch.cuda.is_available = original_cuda_is_available
        torch.cuda.init = original_init
        torch.cuda.is_initialized = original_is_initialized
        torch.cuda.device_count = original_device_count
        torch.cuda.current_device = original_current_device
        torch.cuda.get_device_name = original_get_device_name
        torch.cuda.get_device_capability = original_get_device_capability
        torch.cuda.get_device_properties = original_get_device_properties
        torch.Tensor.cuda = original_cuda # type: ignore[method-assign]
        torch.Tensor.to = original_to # type: ignore[method-assign]

def get_test_audio(
    num_audios: int=1,
    directory: str="sources",
    subject: Optional[str]=None,
    extensions: List[str]=[".wav", ".mp3", ".flac", ".ogg"],
    include_transcript: bool=False,
    shuffle: bool=False,
) -> Union[str, Tuple[str, str], List[str], List[Tuple[str, str]]]:
    """
    Returns a test audio file.
    """
    data_directory = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", # root
            "tests",
            "data",
            directory,
            "audio"
        )
    )
    returned_audios = []
    for root, _, files in os.walk(data_directory):
        if subject is not None:
            files = [file for file in files if file.startswith(subject)]
        files = [file for file in files if any(file.endswith(extension) for extension in extensions)]
        returned_audios.extend([os.path.join(root, file) for file in files])
    if len(returned_audios) < num_audios:
        raise FileNotFoundError(f"Only {len(returned_audios)} audio files found in {data_directory}, but {num_audios} were requested.")

    if shuffle:
        result = choices(returned_audios, k=num_audios)
    else:
        result = returned_audios[:num_audios]

    if include_transcript:
        result_transcript_files = [os.path.splitext(audio)[0] + ".txt" for audio in result]
        result_transcripts = [
            open(file, "r", encoding="utf-8", newline="").read().strip()
            for file in result_transcript_files
        ]
        if num_audios == 1:
            return result[0], result_transcripts[0]
        return list(zip(result, result_transcripts))
    if num_audios == 1:
        return result[0]
    return result

def get_test_images(
    num_images: int=1,
    directory: str="sources",
    size: Optional[str]=None,
    subject: Optional[str]=None,
    number: Optional[int]=None,
    shuffle: bool=False,
) -> List[Image.Image]:
    """
    Returns a test image.
    """
    from PIL import Image
    data_directory = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", # root
            "tests",
            "data",
            directory,
            "images"
        )
    )
    if size is None:
        subject = None
        number = None
    elif subject is None:
        number = None

    if size is not None:
        if "x" in size:
            width, height = size.split("x")
        else:
            width = height = size
        directory = os.path.join(data_directory, width)
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Size {width} not found in test images.")
    else:
        directories = os.listdir(data_directory)
        directory = os.path.join(data_directory, choice(directories))

    found_images = os.listdir(directory)
    if subject is not None:
        found_images = [image for image in found_images if image.startswith(subject)]
    if size is not None:
        found_images = [image for image in found_images if size in image]
    if number is not None:
        found_images = [image for image in found_images if f"{number:04d}" in image]
    if not found_images:
        raise FileNotFoundError(f"No images found in {directory} with the specified criteria.")
    if len(found_images) < num_images:
        raise FileNotFoundError(f"Only {len(found_images)} images found in {directory}, but {num_images} were requested.")

    if shuffle:
        found_images = choices(found_images, k=num_images)
    else:
        found_images = found_images[:num_images]

    return [
        Image.open(os.path.join(directory, image))
        for image in found_images
    ]

def get_test_image(
    directory: str="sources",
    size: Optional[str]=None,
    subject: Optional[str]=None,
    number: Optional[int]=None,
) -> Image.Image:
    """
    Returns a test image.
    """
    return get_test_images(
        num_images=1,
        directory=directory,
        size=size,
        subject=subject,
        number=number
    )[0]

def get_test_results(
    num_images: int=1,
    size: Optional[str]=None,
    subject: Optional[str]=None,
    number: Optional[int]=None,
) -> List[Image.Image]:
    """
    Returns a test image.
    """
    return get_test_images(
        num_images=num_images,
        directory="results",
        size=size,
        subject=subject,
        number=number
    )

def get_test_result(
    size: Optional[str]=None,
    subject: Optional[str]=None,
    number: Optional[int]=None,
) -> Image.Image:
    """
    Returns a test image.
    """
    return get_test_results(
        size=size,
        subject=subject,
        number=number
    )[0]

def save_test_image(
    image: Image.Image,
    subject: str,
    directory: str="results",
) -> None:
    """
    Saves a test image.
    """
    data_directory = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", # root
            "tests",
            "data",
            directory,
            "images"
        )
    )
    width, height = image.size
    size = max(width, height)
    save_directory = os.path.join(data_directory, f"{size}")
    os.makedirs(save_directory, exist_ok=True)
    existing_subject_images = [image for image in os.listdir(save_directory) if subject in image]
    number = len(existing_subject_images) + 1
    image.save(os.path.join(save_directory, f"{subject}_{width}x{height}_{number:04d}.png"))

def save_test_video(
    frames: List[Image.Image],
    subject: str,
    format: str="mp4",
    directory: str="results",
    frame_rate: int=8,
) -> None:
    """
    Saves a test video.
    """
    from .video_util import Video
    data_directory = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", # root
            "tests",
            "data",
            directory,
            "video"
        )
    )
    os.makedirs(data_directory, exist_ok=True)
    existing_subject_videos = [video for video in os.listdir(data_directory) if subject in video]
    number = len(existing_subject_videos) + 1
    Video(
        frames=frames,
        frame_rate=frame_rate
    ).save(os.path.join(data_directory, f"{subject}_{number:04d}.{format}"))

def save_test_audio(
    audio: Union[torch.Tensor, str, bytes, EncodedAudioProxy],
    subject: str,
    directory: str="results",
    sample_rate: int=44100,
    data_format: str="wav",
) -> None:
    """
    Saves a test audio.
    """
    from .audio_util import audio_write, audio_to_bct_tensor
    data_directory = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", # root
            "tests",
            "data",
            directory,
            "audio"
        )
    )
    os.makedirs(data_directory, exist_ok=True)
    existing_subject_audios = [audio for audio in os.listdir(data_directory) if subject in audio]
    number = len(existing_subject_audios) + 1
    if isinstance(audio, str):
        extension = os.path.splitext(audio)[1]
        os.rename(audio, os.path.join(data_directory, f"{subject}_{number:04d}.{extension}"))
    elif isinstance(audio, bytes):
        with open(os.path.join(data_directory, f"{subject}_{number:04d}.{data_format}"), "wb") as file:
            file.write(audio)
    elif isinstance(audio, EncodedAudioProxy):
        with open(os.path.join(data_directory, f"{subject}_{number:04d}.{audio.format}"), "wb") as file:
            file.write(audio.data)
    else:
        audio_write(
            os.path.join(data_directory, f"{subject}_{number:04d}"),
            audio_to_bct_tensor(audio, sample_rate=sample_rate)[0][0],
            sample_rate=sample_rate
        )

IMAGE_SIMILARITY: Optional[Task] = None
def get_image_similarity(
    left: Image.Image,
    right: Image.Image,
    method: Literal["mse", "ssim", "psnr", "histogram", "features"]="mse",
) -> float:
    """
    Returns the similarity between two images.
    """
    global IMAGE_SIMILARITY
    if IMAGE_SIMILARITY is None:
        from ..tasks import Task
        task_class = Task.get("image-similarity", model=None)
        if task_class is None:
            raise ImportError("Image similarity task is not available.")
        IMAGE_SIMILARITY = task_class()
    similarity_score = IMAGE_SIMILARITY(left=left, right=right, method=method)
    return similarity_score # type: ignore[no-any-return]

def assert_image_equality(
    left: Image.Image,
    right: Image.Image,
    method: Literal["mse", "ssim", "psnr", "histogram", "features"]="mse",
    relative_tolerance: float=1e-5,
    absolute_tolerance: float=1e-8,
) -> None:
    """
    Asserts that two images are equal.
    """
    similarity_score = get_image_similarity(left, right, method=method)
    assert_test_output(
        similarity_score,
        1.0,
        relative_tolerance,
        absolute_tolerance
    )

TEXT_SIMILARITY: Optional[Task] = None
def get_text_similarity(
    reference: str,
    hypothesis: str,
    method: Literal["bleu", "jaccard", "cosine", "rouge", "wer", "mer"]="bleu",
    ngram: Optional[int]=None,
) -> float:
    """
    Returns the similarity between two texts.
    """
    global TEXT_SIMILARITY
    if TEXT_SIMILARITY is None:
        from ..tasks import Task
        task_class = Task.get("text-similarity", model=None)
        if task_class is None:
            raise ImportError("Text similarity task is not available.")
        TEXT_SIMILARITY = task_class()

    similarity_score = TEXT_SIMILARITY(
        reference=reference,
        hypothesis=hypothesis,
        method=method,
        ngram=ngram
    )
    return similarity_score # type: ignore[no-any-return]

def assert_text_equality(
    reference: str,
    hypothesis: str,
    method: Literal["bleu", "jaccard", "cosine", "rouge", "wer", "mer"]="bleu",
    ngram: Optional[int]=None,
    relative_tolerance: float=1e-5,
    absolute_tolerance: float=1e-8,
) -> None:
    """
    Asserts that two texts are equal.
    """
    similarity_score = get_text_similarity(
        reference,
        hypothesis,
        method=method,
        ngram=ngram
    )
    assert_test_output(
        similarity_score,
        1.0,
        relative_tolerance,
        absolute_tolerance
    )

def assert_test_output(
    output: Any,
    expected: Any,
    relative_tolerance: float=1e-5,
    absolute_tolerance: float=1e-8,
) -> None:
    """
    Asserts that the output of a test is equal to the expected output.
    """
    if isinstance(output, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(output) == len(expected)
        for o, e in zip(output, expected):
            assert_test_output(o, e, relative_tolerance, absolute_tolerance)
    elif isinstance(output, dict) and isinstance(expected, dict):
        assert set(output.keys()) == set(expected.keys())
        for key in output:
            assert_test_output(output[key], expected[key], relative_tolerance, absolute_tolerance)
    else:
        mro_type_names = [type(output).__name__] + [base.__name__ for base in type(output).__mro__]
        if isinstance(output, (int, float)) and isinstance(expected, (int, float)):
            assert abs(output - expected) <= absolute_tolerance + relative_tolerance * abs(expected)
        elif "Tensor" in mro_type_names:
            assert torch.allclose(output, expected, rtol=relative_tolerance, atol=absolute_tolerance)
        elif "Image" in mro_type_names:
            assert_image_equality(output, expected, relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance)
        elif "ndarray" in mro_type_names:
            from numpy import allclose
            assert allclose(output, expected, rtol=relative_tolerance, atol=absolute_tolerance)
        else:
            assert output == expected

STATIC_MEMORY_TESTED_TASKS: Set[str] = set()
def execute_task_test_suite(
    task_name: str,
    model: Optional[str]=None,
    task_config: Optional[Dict[str, Any]]=None,
    ensure_availability: bool=True,
    num_exercise_executions: int=6,
    assert_static_memory_ratio: Optional[float]=0.25,
    assert_runtime_memory_ratio: Optional[float]=0.25,
    memory_epsilon_gb: float=1e-3, # 1 MB
    first_execution_weight: float=0.05,
    subsequent_execution_weight: float=1.0,
    fail_fast: bool=True,
    use_profiler: bool=False,
    use_import_context: bool=True,
    cases: Optional[List[Tuple[Dict[str, Any], Any]]]=None,
    relative_tolerance: float=1e-5,
    absolute_tolerance: float=1e-8,
) -> List[Any]:
    """
    Executes a task test suite.
    """
    from ..tasks import Task
    from .log_util import logger

    with log_duration("Locating task"):
        task_class = Task.get(task_name, model=model, available_only=False)
        assert task_class is not None
    if not cases:
        warnings.warn("No test cases provided, exiting!")
        return []

    if task_class.get_key() in STATIC_MEMORY_TESTED_TASKS:
        assert_static_memory_ratio = None # These can only really be tested once
    elif assert_static_memory_ratio is not None:
        STATIC_MEMORY_TESTED_TASKS.add(task_class.get_key())

    last_exception: Optional[Exception] = None
    test_output: List[Any] = []
    # First use exercise to ensure the task is available and accurately reports requirements
    for i, (parameters, output) in enumerate(cases):
        try:
            logger.debug(f"Running test case {i + 1}")
            result = task_class.exercise(
                task_config,
                ensure_availability=ensure_availability,
                num_executions=num_exercise_executions,
                assert_static_memory_ratio=assert_static_memory_ratio,
                assert_runtime_memory_ratio=assert_runtime_memory_ratio,
                memory_epsilon_gb=memory_epsilon_gb,
                first_execution_weight=first_execution_weight,
                subsequent_execution_weight=subsequent_execution_weight,
                use_profiler=use_profiler,
                use_import_context=use_import_context,
                **parameters
            )
            test_output.append(result)
            if output is not None:
                assert_test_output(result, output, relative_tolerance, absolute_tolerance)
            assert_static_memory_ratio = None
        except Exception as e:
            if fail_fast:
                raise e
            last_exception = e
            continue

    # Finally, raise the last exception if it exists
    if last_exception is not None and not fail_fast:
        raise last_exception

    return test_output
