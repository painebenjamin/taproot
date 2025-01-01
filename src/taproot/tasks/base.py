from __future__ import annotations

import os
import tempfile
import warnings

from typing import Iterable, Tuple, Any, Dict, Optional, Type, List, Callable, Union, Sequence, TYPE_CHECKING
from typing_extensions import TypedDict, Literal

from time import perf_counter
from contextlib import nullcontext

from ..constants import *
from ..payload import *
from ..util import (
    # Methods
    audio_write,
    check_download_files_to_dir,
    file_is_downloaded_to_dir,
    estimate_parameter_bytes,
    generate_id,
    get_combined_specifications,
    get_file_name_from_url,
    get_metadata,
    get_optimal_device,
    get_test_restricted_import_context,
    get_torch_dtype,
    human_duration,
    human_size,
    install_packages,
    installed_package_matches_spec,
    is_numpy_array,
    is_torch_tensor,
    log_duration,
    logger,
    profiler,
    ram_counter,
    restrict_gpu,
    time_counter,
    timed_lru_cache,
    to_bchw_tensor,
    to_bhwc_ndarray,
    to_jpeg_array,
    to_pil_array,
    trim_silence,
    vram_counter,
    # Classes
    EncodedAudioProxy,
    EncodedVideoProxy,
    EncodedImageProxy,
    PretrainedModelMixin,
    MachineCapability,
    GPU,
    IntrospectableMixin,
    CallSignature,
    Video,
)
from ..config import ConfigMixin, TaskConfig

from .util import PretrainedLoader, TaskLoader

if TYPE_CHECKING:
    import torch
    import numpy as np
    from turbojpeg import TurboJPEG # type: ignore[import-not-found,import-untyped,unused-ignore]

    from ..hinting import ImageType, AudioType, AudioResultType

__all__ = [
    "Task",
    "ExecutableTask",
    "ExecutableTaskModel",
]

class ExecutableTaskModel(CallSignature):
    """
    Model of a task that can be executed by a server.
    """
    task: Type[Task]

class ExecutableTask(TypedDict):
    """
    Tasks (roots) that can be executed by a server.
    """
    models: Dict[Optional[str], ExecutableTaskModel]
    default: Optional[str]

class Task(ConfigMixin, IntrospectableMixin):
    """
    An executable task.
    """
    config_class = TaskConfig

    """Global metadata"""
    task: str  # Task name
    model: Optional[str] = None  # Model name, must be unique with task
    default: bool = False  # Whether this is the default model for the task

    """Optional task metadata"""
    use_gpu: bool = False
    load_on_gpu: Optional[bool] = None
    gpu_precision: Optional[str] = None
    cuda_version: Optional[str] = None
    static_memory_gb: Optional[float] = None
    runtime_memory_gb: Optional[float] = None
    static_gpu_memory_gb: Optional[float] = None
    runtime_gpu_memory_gb: Optional[float] = None

    """Catalog metadata"""
    display_name: Optional[str] = None  # Display name for the task

    """Authorship metadata"""
    author: Optional[str] = None # Author of the components needed for the task
    author_url: Optional[str] = None # URL to author's website or profile
    author_additional: Optional[List[str]] = None # Additional authors of the components needed for the task
    author_affiliations: Optional[List[str]] = None # Affiliations of the authors of the components needed for the task
    author_journal: Optional[str] = None # Journal or conference where the components needed for the task were published
    author_journal_pages: Optional[str] = None # Pages in the journal or conference where the components needed for the task were published
    author_journal_volume: Optional[str] = None # Volume of the journal or conference where the components needed for the task were published
    author_journal_title: Optional[str] = None # Title of the publication in the journal or conference where the components needed for the task were published
    author_journal_year: Optional[int] = None # Year of publication of the components needed for the task
    finetune_author: Optional[str] = None # Author of the finetuned model
    finetune_author_url: Optional[str] = None # URL to author's website or profile
    implementation_author: Optional[str] = None # Author of the implementation if different from paper/journal authors
    implementation_author_url: Optional[str] = None # URL to author's website or profile

    """License metadata"""
    license: Optional[str] = None  # License for the components needed for the task
    license_url: Optional[str] = None  # URL to license details if not a standard license
    license_attribution: Optional[bool] = None  # Whether the license requires attribution
    license_derivatives: Optional[bool] = None  # Whether the license allows derivatives
    license_redistribution: Optional[bool] = None  # Whether the license allows redistribution
    license_copy_left: Optional[bool] = None  # Whether the license is a copyleft license
    license_commercial: Optional[bool] = None  # Whether the license allows commercial use
    license_hosting: Optional[bool] = None  # Whether the license allows hosting

    """Testing metadata"""
    measure_memory: bool = True
    measure_gpu_memory: bool = True
    measure_nvidia_smi: bool = False

    """Components""" 
    pretrained_models: Optional[Dict[str, Type[PretrainedModelMixin]]] = None
    offload_models: Optional[Union[List[str], bool]] = None
    component_tasks: Optional[Dict[str, Type[Task]]] = None
    offload_tasks: Optional[Union[List[str], bool]] = None

    """Optional components"""
    optional_pretrained_models: Optional[Dict[str, Type[PretrainedModelMixin]]] = None
    optional_component_tasks: Optional[Dict[str, Type[Task]]] = None

    """Private properties"""
    _dtype: Optional[torch.dtype]
    _is_available: bool
    _start: float = 0.0
    _step: int = 0
    _num_steps: int = 1
    _step_update: float = 0.0
    _rate_ema: float = 0.0
    _intermediates: List[Any] = []

    """Task properties"""

    @property
    def gpu_index(self) -> int:
        """
        Get the GPU index.
        """
        if self.config.gpu_index is None:
            return 0
        return int(self.config.gpu_index)

    @gpu_index.setter
    def gpu_index(self, value: int) -> None:
        """
        Set the GPU index.
        """
        if value < 0:
            raise ValueError("GPU index must be non-negative.")
        self.config.gpu_index = value
        del self.device

    @property
    def device(self) -> torch.device:
        """
        Get the device.
        """
        if not hasattr(self, "_device"):
            if self.gpu_index < 0:
                raise ValueError("GPU index must be non-negative.")
            self._device = get_optimal_device(self.gpu_index)
        return self._device

    @device.deleter
    def device(self) -> None:
        """
        Delete the device.
        """
        if hasattr(self, "_device"):
            del self._device

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """
        Get the data type.
        """
        if not hasattr(self, "_dtype"):
            configured_dtype = self.config.dtype
            if configured_dtype is not None:
                try:
                    self._dtype = get_torch_dtype(configured_dtype)
                except ValueError:
                    self._dtype = None
            else:
                class_dtype = self.required_gpu_precision()
                if class_dtype is not None:
                    self._dtype = get_torch_dtype(class_dtype)
                else:
                    self._dtype = None
        return self._dtype

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        """
        Set the data type.
        """
        self.config.dtype = str(value)
        del self.dtype

    @dtype.deleter
    def dtype(self) -> None:
        """
        Delete the data type.
        """
        if hasattr(self, "_dtype"):
            del self._dtype

    @property
    def model_dir(self) -> str:
        """
        Get the model directory.
        """
        if self.config.model_dir is None:
            return DEFAULT_MODEL_DIR
        return str(self.config.model_dir)

    @model_dir.setter
    def model_dir(self, value: str) -> None:
        """
        Set the model directory.
        """
        self.config.model_dir = value

    @property
    def rate_ema_alpha(self) -> float:
        """
        Get the rate EMA alpha.
        """
        if self.config.rate_ema_alpha is None:
            return STEP_RATE_EMA_ALPHA
        return float(self.config.rate_ema_alpha)

    @rate_ema_alpha.setter
    def rate_ema_alpha(self, value: float) -> None:
        """
        Set the rate EMA alpha.
        """
        self.config.rate_ema_alpha = value

    @property
    def use_tqdm(self) -> bool:
        """
        Get whether to use tqdm.
        """
        return bool(self.config.use_tqdm)

    @use_tqdm.setter
    def use_tqdm(self, value: bool) -> None:
        """
        Set whether to use tqdm.
        """
        self.config.use_tqdm = value

    @property
    def allow_optional(self) -> bool:
        """
        Get whether to allow optional components.
        """
        return bool(self.config.allow_optional)

    @allow_optional.setter
    def allow_optional(self, value: bool) -> None:
        """
        Set whether to allow optional components.
        """
        previous_value = self.config.allow_optional
        if previous_value == value:
            return

        self.config.allow_optional = value

        if value:
            # Add optional components to loaders, if they exist
            if getattr(self, "_pretrained", None) is not None and self.optional_pretrained_models is not None:
                self._pretrained.models.update(self.optional_pretrained_models)
            if getattr(self, "_tasks", None) is not None and self.optional_component_tasks is not None:
                self._tasks.tasks.update(self.optional_component_tasks)
        else:
            # Remove optional components from loaders, if they exist
            if getattr(self, "_pretrained", None) is not None and self.optional_pretrained_models is not None:
                for model_name in self.optional_pretrained_models.keys():
                    self._pretrained.unload_by_name(model_name)
                    self._pretrained.models.pop(model_name, None)
            if getattr(self, "_tasks", None) is not None and self.optional_component_tasks is not None:
                for task_name in self.optional_component_tasks.keys():
                    self._tasks.unload_by_name(task_name)
                    self._tasks.tasks.pop(task_name, None)

    @property
    def rate(self) -> float:
        """
        Get the rate.
        """
        return self._rate_ema

    @property
    def num_steps(self) -> int:
        """
        Get the number of steps.
        """
        return self._num_steps

    @num_steps.setter
    def num_steps(self, value: int) -> None:
        """
        Set the number of steps.
        Reset the step counter.
        """
        self._step = 0
        self._start = perf_counter()
        self._num_steps = value
        self._step_update = self._start
        self._rate_ema = 0.0
        self._intermediates = []

    @property
    def step(self) -> int:
        """
        Get the step.
        """
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        """
        Set the step and update the rate.
        """
        set_time = perf_counter()
        step_delta = value - self._step
        time_delta = set_time - self._step_update

        self._step = value
        self._step_update = set_time
        if self._rate_ema == 0.0:
            self._rate_ema = step_delta / time_delta
        else:
            self._rate_ema = self._rate_ema * (1 - self.rate_ema_alpha) + step_delta / time_delta * self.rate_ema_alpha

    @property
    def save_dir(self) -> str:
        """
        Get the save directory.
        """
        config_save_dir = self.config.save_dir
        if config_save_dir is None:
            return self.default_save_dir
        return str(self.config.save_dir)

    @save_dir.setter
    def save_dir(self, value: Optional[str]) -> None:
        """
        Set the save directory.
        """
        self.config.save_dir = value

    @property
    def enable_model_offload(self) -> bool:
        """
        Get whether to enable model offloading.
        """
        return bool(self.config.enable_model_offload)

    @enable_model_offload.setter
    def enable_model_offload(self, value: bool) -> None:
        """
        Set whether to enable model offloading.
        """
        self.config.enable_model_offload = value

    @property
    def enable_sequential_offload(self) -> bool:
        """
        Get whether to enable sequential offloading.
        """
        return bool(self.config.enable_sequential_offload)

    @enable_sequential_offload.setter
    def enable_sequential_offload(self, value: bool) -> None:
        """
        Set whether to enable sequential offloading.
        """
        self.config.enable_sequential_offload = value

    @property
    def enable_encode_tiling(self) -> bool:
        """
        Get whether to enable tiled encoding.
        """
        return bool(self.config.enable_encode_tiling)

    @enable_encode_tiling.setter
    def enable_encode_tiling(self, value: bool) -> None:
        """
        Set whether to enable tiled encoding.
        """
        self.config.enable_encode_tiling = value

    @property
    def enable_encode_slicing(self) -> bool:
        """
        Get whether to enable sliced encoding.
        """
        return bool(self.config.enable_encode_slicing)

    @enable_encode_slicing.setter
    def enable_encode_slicing(self, value: bool) -> None:
        """
        Set whether to enable sliced encoding.
        """
        self.config.enable_encode_slicing = value

    @property
    def context_length(self) -> Optional[int]:
        """
        Get the context length.
        """
        if self.config.context_length is None:
            return None
        return int(self.config.context_length)

    @context_length.setter
    def context_length(self, value: Optional[int]) -> None:
        """
        Set the context length.
        """
        self.config.context_length = value

    """Getters"""

    @property
    def default_save_dir(self) -> str:
        """
        The default save directory for images.
        """
        return os.getcwd()

    @property
    def intermediates(self) -> List[Any]:
        """
        Get the intermediates.
        """
        return self._intermediates

    @property
    def last_intermediate(self) -> Any:
        """
        Gets the last intermediate.
        """
        if not self._intermediates:
            return None
        return self._intermediates[-1]

    @property
    def step_iterator(self) -> Iterable[int]:
        """
        Iterate over the steps.
        """
        for i in range(self.num_steps):
            yield self.step
            self.increment_step()

    @property
    def jpeg_encoder(self) -> TurboJPEG:
        """
        Get the JPEG encoder using TurboJPEG.
        Will raise an ImportError if TurboJPEG is not installed.
        """
        if not hasattr(self, "_jpeg"):
            from turbojpeg import TurboJPEG
            self._jpeg = TurboJPEG()
        return self._jpeg

    @property
    def gpu(self) -> GPU:
        """
        Get the GPU specification.
        """
        capability = self.get_capability()
        try:
            return capability.gpus[self.gpu_index] # type: ignore[no-any-return]
        except IndexError:
            raise RuntimeError(f"GPU index {self.gpu_index} not available!")

    @property
    def options(self) -> Dict[str, Any]:
        """
        Get the options that don't otherwise fit.
        """
        return self.config.options # type: ignore[no-any-return]

    """Calculated properties"""

    @property
    def progress(self) -> float:
        """
        Gets progress from 0 to 1.
        """
        return self._step / self._num_steps

    @property
    def elapsed(self) -> float:
        """
        Gets the elapsed time.
        """
        return perf_counter() - self._start

    @property
    def remaining(self) -> float:
        """
        Gets an estimate of the remaining time.
        """
        if self._rate_ema == 0.0:
            return float("inf")
        return (self._num_steps - self._step) / self._rate_ema

    @property
    def pretrained(self) -> PretrainedLoader:
        """
        Gets the pretrained loader.
        Instantiates a new one if it doesn't exist.
        """
        if not hasattr(self, "_pretrained"):
            use_gpu = self.requires_gpu()
            self._pretrained = self.get_pretrained_loader(
                self.model_dir,
                device=None if not use_gpu or self.load_on_gpu == False else self.device,
                dtype=None if not use_gpu or self.load_on_gpu == False else self.dtype,
                allow_optional=self.allow_optional,
            )
        return self._pretrained

    @property
    def tasks(self) -> TaskLoader:
        """
        Gets the task loader.
        """
        if not hasattr(self, "_tasks"):
            use_gpu = self.requires_gpu()
            self._tasks = self.get_task_loader(
                self.model_dir,
                device=None if not use_gpu or self.load_on_gpu == False else self.device,
                dtype=None if not use_gpu or self.load_on_gpu == False else self.dtype,
                allow_optional=self.allow_optional,
            )
        return self._tasks

    """Task classmethods"""

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Get the required packages for the task.
        Should be a dictionary of package names and versions.
        Default behavior checks for GPU requirements and returns
        the default set for that.
        """
        packages: Dict[str, Optional[str]] = {}
        if cls.requires_gpu():
            packages = {
                "pil": PILLOW_VERSION_SPEC,
                "numpy": NUMPY_VERSION_SPEC,
                "torch": TORCH_VERSION_SPEC
            }
            if any([
                os.path.splitext(file)[1] == ".safetensors"
                for file in cls.required_files()
            ]):
                packages["safetensors"] = SAFETENSORS_VERSION_SPEC
        return packages

    @classmethod
    def required_files(cls, allow_optional: bool=True) -> List[str]:
        """
        Get the required files for the task.
        Should be URLs to download the files.
        Default behavior is to look at any configured pretrained models.
        """
        return cls.get_pretrained_loader(allow_optional=allow_optional).get_required_files() + \
               cls.get_task_loader(allow_optional=allow_optional).get_required_files()

    @classmethod
    def required_gpu_model_files(cls) -> List[str]:
        """
        Get the required GPU model files for the task.
        A filtered view of required files.
        """
        return [
            file
            for file in cls.required_files()
            if os.path.splitext(file)[1] in KNOWN_GPU_MODEL_FILE_EXTENSIONS
        ]

    @classmethod
    def combined_required_packages(cls, allow_optional: bool=True) -> Dict[str, Optional[str]]:
        """
        Get the combined required packages for the task.
        """
        # First check for required packages for this task
        packages: Dict[str, List[Optional[str]]] = {}
        for package, version in cls.required_packages().items():
            if package not in packages:
                packages[package] = []
            packages[package].append(version)

        # Now add requirements for any component tasks
        if cls.component_tasks is not None:
            for sub_task, sub_task_class in cls.component_tasks.items():
                for package, version in sub_task_class.required_packages().items():
                    if package not in packages:
                        packages[package] = []
                    packages[package].append(version)

        # Also include optional component tasks if allowed
        if allow_optional and cls.optional_component_tasks is not None:
            for sub_task, sub_task_class in cls.optional_component_tasks.items():
                for package, version in sub_task_class.required_packages().items():
                    if package not in packages:
                        packages[package] = []
                    packages[package].append(version)

        # Combine the versions
        return dict([
            (package, get_combined_specifications(*versions))
            for package, versions in packages.items()
        ])

    @classmethod
    def is_available(
        cls,
        model_dir: str=DEFAULT_MODEL_DIR,
        allow_optional: bool=False
    ) -> bool:
        """
        Check if the task is available.
        """
        # First check for required packages for this task
        for package, version in cls.combined_required_packages(allow_optional=allow_optional).items():
            if not installed_package_matches_spec(package, version):
                return False

        # Next check for required files
        for file in cls.required_files(allow_optional=allow_optional):
            if not file_is_downloaded_to_dir(
                file,
                model_dir,
                check_size=False
            ):
                return False
        return True

    @classmethod
    def get_pending_downloads(
        cls,
        model_dir: str=DEFAULT_MODEL_DIR,
        allow_optional: bool=True,
    ) -> List[str]:
        """
        Get the pending downloads for the task.
        """
        pending_downloads = []
        for file in cls.required_files(allow_optional=allow_optional):
            if not file_is_downloaded_to_dir(file, model_dir):
                pending_downloads.append(file)
        return pending_downloads

    @classmethod
    def get_pending_packages(
        cls,
        allow_optional: bool=True,
    ) -> Dict[str, Optional[str]]:
        """
        Get the pending packages for the task.
        """
        pending_packages = {}
        for package, version in cls.combined_required_packages(allow_optional=allow_optional).items():
            if not installed_package_matches_spec(package, version):
                pending_packages[package] = version
        return pending_packages

    @classmethod
    def get_pretrained_loader(
        cls,
        model_dir: str=DEFAULT_MODEL_DIR,
        device: Optional[Union[str, torch.device]]=None,
        dtype: Optional[Union[str, torch.dtype]]=None,
        allow_optional: bool=True,
    ) -> PretrainedLoader:
        """
        Get the pretrained loader for the task.
        """
        return PretrainedLoader(
            model_dir,
            device=device,
            dtype=dtype,
            **{
                **({} if cls.pretrained_models is None else cls.pretrained_models),
                **({} if cls.optional_pretrained_models is None or not allow_optional else cls.optional_pretrained_models)
            }
        )

    @classmethod
    def get_task_loader(
        cls,
        model_dir: str=DEFAULT_MODEL_DIR,
        device: Optional[Union[str, torch.device]]=None,
        dtype: Optional[Union[str, torch.dtype]]=None,
        allow_optional: bool=True,
    ) -> TaskLoader:
        """
        Get the task loader for components of the task.
        """
        return TaskLoader(
            model_dir,
            device=device,
            dtype=dtype,
            **{
                **({} if cls.component_tasks is None else cls.component_tasks),
                **({} if cls.optional_component_tasks is None or not allow_optional else cls.optional_component_tasks)
            }
        )

    @classmethod
    def download_required_files(
        cls,
        model_dir: str=DEFAULT_MODEL_DIR,
        chunk_size: int=8192,
        check_size: bool=True,
        progress_callback: Optional[Callable[[int, int, int, int], None]]=None,
        text_callback: Optional[Callable[[str], None]]=None,
        authorization: Optional[str]=None,
    ) -> None:
        """
        Download the required files for the task.
        """
        required_files = cls.get_pending_downloads(model_dir)
        check_download_files_to_dir(
            required_files,
            model_dir,
            chunk_size=chunk_size,
            check_size=check_size,
            progress_callback=progress_callback,
            text_callback=text_callback,
            authorization=authorization,
        )

    @classmethod
    def install_required_packages(cls) -> None:
        """
        Install the required packages for the task.
        """
        install_packages(cls.get_pending_packages())

    @classmethod
    def ensure_availability(
        cls,
        model_dir: str=DEFAULT_MODEL_DIR,
        chunk_size: int=8192,
        check_size: bool=True,
        progress_callback: Optional[Callable[[int, int, int, int], None]]=None,
        text_callback: Optional[Callable[[str], None]]=None,
        authorization: Optional[str]=None,
    ) -> None:
        """
        Ensure that the task is available.
        """
        cls.install_required_packages()
        cls.download_required_files(
            model_dir=model_dir,
            chunk_size=chunk_size,
            check_size=check_size,
            progress_callback=progress_callback,
            text_callback=text_callback,
            authorization=authorization,
        )

    @classmethod
    def requires_compilation(cls) -> bool:
        """
        Optionally specify whether the task requires compilation via torch.compile().
        """
        return False

    @classmethod
    def requires_gpu(cls) -> bool:
        """
        Optionally specify whether the task should use the GPU.
        """
        return bool(cls.required_gpu_model_files()) or cls.use_gpu

    @classmethod
    def required_gpu_precision(cls) -> Optional[str]:
        """
        Optionally specify the GPU precision for the task.
        """
        if cls.gpu_precision is not None:
            return cls.gpu_precision
        if any(
            "fp16" in model_file
            for model_file in cls.required_gpu_model_files()
        ):
            return "half"
        return None

    @classmethod
    def required_cuda_version(cls) -> Optional[str]:
        """
        Optionally specify the required CUDA version for the task.
        """
        return cls.cuda_version

    @classmethod
    def required_static_memory_gb(cls) -> Optional[float]:
        """
        Optionally specify the required static memory for the task.
        """
        return cls.static_memory_gb

    @classmethod
    def combined_required_static_memory_gb(cls) -> Optional[float]:
        """
        Optionally specify the combined required static memory for the task.
        """
        static_memory_gb = [cls.required_static_memory_gb()]
        if cls.component_tasks is not None:
            for task_class in cls.component_tasks.values():
                static_memory_gb.append(task_class.required_static_memory_gb())
        non_none_static_memory_gb = [memory_gb for memory_gb in static_memory_gb if memory_gb is not None]
        if len(non_none_static_memory_gb) == 0:
            return None
        return sum(non_none_static_memory_gb)

    @classmethod
    def required_runtime_memory_gb(cls, **parameters: ParameterMetadataPayload) -> Optional[float]:
        """
        Optionally specify the required memory for the task.
        The default assumes that the parameter size represents the memory usage.
        """
        parameter_gb = estimate_parameter_bytes(**parameters) / 1000 ** 3
        return parameter_gb + (0 if cls.runtime_memory_gb is None else cls.runtime_memory_gb)

    @classmethod
    def required_static_gpu_memory_gb(cls) -> Optional[float]:
        """
        Optionally specify the required GPU memory for the task.
        """
        return cls.static_gpu_memory_gb

    @classmethod
    def combined_required_static_gpu_memory_gb(cls) -> Optional[float]:
        """
        Optionally specify the combined required static memory for the task.
        """
        static_memory_gb = [cls.required_static_gpu_memory_gb()]
        if cls.component_tasks is not None:
            for task_class in cls.component_tasks.values():
                static_memory_gb.append(task_class.required_static_gpu_memory_gb())
        non_none_static_memory_gb = [memory_gb for memory_gb in static_memory_gb if memory_gb is not None]
        if len(non_none_static_memory_gb) == 0:
            return None
        return sum(non_none_static_memory_gb)

    @classmethod
    def required_runtime_gpu_memory_gb(cls, **parameters: ParameterMetadataPayload) -> Optional[float]:
        """
        Optionally specify the required GPU memory for the task.
        """
        parameter_gb = estimate_parameter_bytes(**parameters) / 1000 ** 3
        return parameter_gb + (0 if cls.runtime_gpu_memory_gb is None else cls.runtime_gpu_memory_gb)

    @classmethod
    @timed_lru_cache(ttl=1.0) # 1 second cache
    def get_capability(cls) -> MachineCapability:
        """
        Get the capability of the machine.
        """
        return MachineCapability.get_capability(fail_on_gpu_error=False)

    @classmethod
    def get_display_name(cls) -> str:
        """
        Get the display name.
        """
        return cls.display_name or cls.__name__

    @classmethod
    def get_author_citation(cls) -> str:
        """
        Get the author citation.
        """
        author_string = ""

        if cls.author:
            author_string += cls.author
            if cls.author_additional is not None:
                if len(cls.author_additional) == 1:
                   author_string += f"and {cls.author_additional[0]}"
                else:
                    author_string += f", {', '.join(cls.author_additional[:-1])} and {cls.author_additional[-1]}"

        if cls.author_affiliations:
            if author_string:
                author_string += "\n"
            if len(cls.author_affiliations) == 1:
                author_string += f"{cls.author_affiliations[0]}"
            else:
                author_string += f"{', '.join(cls.author_affiliations[:-1])} and {cls.author_affiliations[-1]}"

        if cls.author_journal:
            if author_string:
                author_string += "\n"
            author_string += f"Published in {cls.author_journal}"
            if cls.author_journal_volume:
                author_string += f", vol. {cls.author_journal_volume}"
            if cls.author_journal_pages:
                author_string += f", pp. {cls.author_journal_pages}"
            if cls.author_journal_title:
                author_string += f", “{cls.author_journal_title}”"
            if cls.author_journal_year:
                author_string += f", {cls.author_journal_year}"

        if cls.author_url:
            if author_string:
                author_string += "\n"
            author_string += cls.author_url

        if cls.finetune_author:
            if author_string:
                author_string += "\n"
            author_string += f"Finetuned by {cls.finetune_author}"
            if cls.finetune_author_url:
                author_string += f" ({cls.finetune_author_url})"

        if cls.implementation_author:
            if author_string:
                author_string += "\n"
            author_string += f"Implementation by {cls.implementation_author}"
            if cls.implementation_author_url:
                author_string += f" ({cls.implementation_author_url})"

        return author_string

    @classmethod
    def get_license_citation(cls) -> str:
        """
        Get the license citation.
        """
        license_string = ""

        if cls.license:
            if cls.license in LICENSE_NAMES:
                license_string += LICENSE_NAMES[cls.license]
            else:
                license_string += cls.license
            if cls.license_url:
                license_string += f" ({cls.license_url})"
            elif cls.license in LICENSE_URLS:
                license_string += f" ({LICENSE_URLS[cls.license]})"
        return license_string

    @classmethod
    def get_license_allowances(
        cls,
        allowed_character: str="✅",
        disallowed_character: str="❌",
        delimiter: str="\n",
    ) -> str:
        """
        Get the license allowances.
        """
        if cls.license and cls.license in LICENSE_ALLOWANCES:
            (
                default_attribution,
                default_derivatives,
                default_redistribution,
                default_copy_left,
                default_commercial,
                default_hosting
            ) = LICENSE_ALLOWANCES[cls.license]
        else:
            default_attribution = True
            default_derivatives = False
            default_redistribution = True # This is true for all models we distribute by their nature
            default_copy_left = False
            default_commercial = False
            default_hosting = False

        attribution = cls.license_attribution if cls.license_attribution is not None else default_attribution
        derivatives = cls.license_derivatives if cls.license_derivatives is not None else default_derivatives
        redistribution = cls.license_redistribution if cls.license_redistribution is not None else default_redistribution
        copy_left = cls.license_copy_left if cls.license_copy_left is not None else default_copy_left
        commercial = cls.license_commercial if cls.license_commercial is not None else default_commercial
        hosting = cls.license_hosting if cls.license_hosting is not None else default_hosting

        return delimiter.join([
            f"{allowed_character if attribution else disallowed_character} Attribution Required",
            f"{allowed_character if derivatives else disallowed_character} Derivatives Allowed",
            f"{allowed_character if redistribution else disallowed_character} Redistribution Allowed",
            f"{allowed_character if copy_left else disallowed_character} Copyleft (Share-Alike) Required",
            f"{allowed_character if commercial else disallowed_character} Commercial Use Allowed",
            f"{allowed_character if hosting else disallowed_character} Hosting Allowed",
        ])

    @classmethod
    def enumerate(
        cls,
        available_only: bool = True,
        model_dir: str = DEFAULT_MODEL_DIR,
        task_model_dirs: Dict[str, str] = {},
    ) -> Iterable[Tuple[str, Optional[str], Type[Task]]]:
        """
        Enumerate all available tasks by task name and model.
        """
        for task_class in cls.__subclasses__():
            task = getattr(task_class, "task", None)
            model = getattr(task_class, "model", None)

            if task:
                task_key = task_class.get_key()
                if task_key in task_model_dirs:
                    task_model_dir = task_model_dirs[task_key]
                elif task in task_model_dirs:
                    task_model_dir = task_model_dirs[task]
                else:
                    task_model_dir = model_dir
                if not available_only or task_class.is_available(model_dir=task_model_dir):
                    yield (task, model, task_class)

            for subtask in task_class.enumerate(
                available_only=available_only,
                model_dir=model_dir,
                task_model_dirs=task_model_dirs,
            ):
                yield subtask

    @classmethod
    def catalog(
        cls,
        available_only: bool = True,
        model_dir: str = DEFAULT_MODEL_DIR,
        task_model_dirs: Dict[str, str] = {},
    ) -> Dict[str, ExecutableTask]:
        """
        Catalog of tasks that can be executed by the server.
        """
        catalog: Dict[str, ExecutableTask] = {}
        for task, model, subcls in cls.enumerate(
            model_dir=model_dir,
            available_only=available_only,
            task_model_dirs=task_model_dirs,
        ):
            if task not in catalog:
                catalog[task] = {"models": {}, "default": None}
            if model in catalog[task]["models"]:
                warnings.warn(f"Duplicate model {model} for task {task}, ignoring.")
                continue
            catalog[task]["models"][model] = {
                "task": subcls,
                **subcls.introspect(),
            }
            if subcls.default:
                catalog[task]["default"] = model
        return catalog

    @classmethod
    def get(
        cls,
        task: str,
        model: Optional[str] = None,
        available_only: bool = True,
        model_dir: str = DEFAULT_MODEL_DIR,
        task_model_dirs: Dict[str, str] = {},
    ) -> Optional[Type[Task]]:
        """
        Get a task by task name and model.
        """
        task_first_class: Optional[Type[Task]] = None
        task_default_class: Optional[Type[Task]] = None
        for task_name, task_model, task_class in cls.enumerate(
            model_dir=model_dir,
            available_only=available_only,
            task_model_dirs=task_model_dirs,
        ):
            if task_name == task:
                if (model is None and task_model is None) or model == task_model:
                    return task_class
                elif task_class.default:
                    task_default_class = task_class
                if task_first_class is None:
                    task_first_class = task_class
        if model is not None:
            return None # Specific model requested but not found
        elif task_default_class is not None:
            return task_default_class # Default model found
        return task_first_class # May be None if no available tasks

    @classmethod
    def get_key(cls) -> str:
        """
        Get the task key.
        """
        task_key = cls.task
        if cls.model is None:
            task_key = f"{task_key}:none"
        else:
            task_key = f"{task_key}:{cls.model}"
        return task_key

    @classmethod
    def exercise(
        cls,
        config: Optional[Dict[str, Any]]=None,
        *,
        ensure_availability: bool=True,
        use_profiler: bool=False,
        use_import_context: bool=True,
        assert_static_memory_ratio: Optional[float]=1.0,
        assert_runtime_memory_ratio: Optional[float]=1.0,
        num_executions: int=5,
        memory_epsilon_gb: float=1e-3, # 1 Mb
        first_execution_weight: float=0.05,
        subsequent_execution_weight: float=1.0,
        **payload: Any,
    ) -> Any:
        """
        Exercise the task.
        """
        with log_duration("Availability check"):
            if not cls.is_available():
                if ensure_availability:
                    cls.ensure_availability()
                    if not cls.is_available():
                        raise RuntimeError(
                            f"Task {cls.task} is not available after calling {cls.__name__}.ensure_availability()."
                        )
                else:
                    raise RuntimeError(
                        f"Task {cls.task} is not available. Before exercising the task, " +
                        "ensure that it is available by calling {cls.__name__}.ensure_availability()."
                    )

        with log_duration("Requirements gathering"):
            requires_gpu = cls.requires_gpu()
            required_static_memory_gb = cls.required_static_memory_gb()
            required_static_gpu_memory_gb = None if not requires_gpu else cls.required_static_gpu_memory_gb()
            required_packages = cls.required_packages()

            gpu_context = restrict_gpu() if not requires_gpu else nullcontext()
            profiler_context = profiler() if use_profiler else nullcontext()
            import_context = lambda: get_test_restricted_import_context(*required_packages) if use_import_context else nullcontext()
            get_ram_counter = ram_counter if cls.measure_memory else nullcontext
            get_vram_counter = lambda: vram_counter(use_nvidia_smi=cls.measure_nvidia_smi) if requires_gpu else nullcontext()

            metadata = get_metadata(payload)
            required_runtime_memory_gb = cls.required_runtime_memory_gb(**metadata)
            required_runtime_gpu_memory_gb = None if not requires_gpu else cls.required_runtime_gpu_memory_gb(**metadata)

            logger.debug(f"Task {cls.__name__} requirements gathered.")
            logger.debug(f"Requires GPU: {requires_gpu}")
            logger.debug(f"Required static memory: {required_static_memory_gb}")
            logger.debug(f"Required runtime memory: {required_runtime_memory_gb}")
            logger.debug(f"Required static GPU memory: {required_static_gpu_memory_gb}")
            logger.debug(f"Required runtime GPU memory: {required_runtime_gpu_memory_gb}")

        execution_memory_bytes: List[int] = []
        execution_gpu_memory_bytes: List[int] = []
        execution_time_seconds: List[float] = []

        def get_weighted_average(arr: Sequence[Union[int, float]]) -> Union[int, float]:
            """
            Get the weighted average of an array.
            """
            if len(arr) == 0:
                return 0
            if len(arr) == 1:
                return arr[0]
            weight_sum = first_execution_weight + sum(subsequent_execution_weight for _ in arr[1:])
            return (first_execution_weight * arr[0] + sum(subsequent_execution_weight * x for x in arr[1:])) / weight_sum

        logger.debug("Initializing testing context.")
        with gpu_context:
            with profiler_context:
                with import_context(): # type: ignore[no-untyped-call]
                    with get_ram_counter() as static_ram_usage:
                        with get_vram_counter() as static_vram_usage: # type: ignore[no-untyped-call]
                            with time_counter() as initialization_time:
                                logger.debug("Test context initialized, creating class.")
                                task = cls(config)
                            with time_counter() as load_time:
                                logger.debug("Loading class resources.")
                                task.load()
                    for i in range(num_executions):
                        with get_ram_counter() as runtime_ram_usage:
                            with get_vram_counter() as runtime_vram_usage: # type: ignore[no-untyped-call]
                                with time_counter() as execution_time_tracker:
                                    task.num_steps = 1
                                    logger.debug(f"Starting execution {i + 1}.")
                                    result = task(**payload)
                        if runtime_ram_usage is None:
                            execution_memory_bytes.append(0)
                        else:
                            execution_memory_bytes.append(int(runtime_ram_usage))
                        execution_time_seconds.append(float(execution_time_tracker))
                        if requires_gpu:
                            if runtime_vram_usage is None:
                                execution_gpu_memory_bytes.append(0)
                            else:
                                execution_gpu_memory_bytes.append(int(runtime_vram_usage))

        # Unload
        task.unload()
        execution_time = get_weighted_average(execution_time_seconds)
        if static_ram_usage is None:
            static_ram_usage_bytes = 0
        else:
            static_ram_usage_bytes = int(static_ram_usage)
        runtime_ram_usage_bytes = get_weighted_average(execution_memory_bytes)
        if requires_gpu:
            if static_vram_usage is None:
                static_vram_usage_bytes = 0
            else:
                static_vram_usage_bytes = int(static_vram_usage)
            runtime_vram_usage_bytes = get_weighted_average(execution_gpu_memory_bytes)

        logger.info(f"{cls.__name__} exercised successfully.")
        logger.info(f"Initialization time: {human_duration(float(initialization_time))}")
        logger.info(f"Load time: {human_duration(float(load_time))}")
        logger.info(f"Execution time average: {human_duration(float(execution_time))}")
        logger.info(f"Execution times: " + ", ".join([human_duration(float(time)) for time in execution_time_seconds]))
        logger.info(f"Static RAM usage: {human_size(static_ram_usage_bytes)}")
        logger.info(f"Runtime RAM usage average: {human_size(runtime_ram_usage_bytes)}")
        logger.info(f"Runtime RAM usages: " + ", ".join([human_size(memory) for memory in execution_memory_bytes]))

        if requires_gpu:
            logger.info(f"Static VRAM usage: {human_size(static_vram_usage_bytes)}")
            logger.info(f"Runtime VRAM usage average: {human_size(runtime_vram_usage_bytes)}")
            logger.info(f"Runtime VRAM usages: " + ", ".join([human_size(memory) for memory in execution_gpu_memory_bytes]))

        # Memory assertions
        failures: List[str] = []

        if assert_static_memory_ratio is not None and cls.measure_memory:
            if required_static_memory_gb is None:
                required_static_memory_gb = 0
            lower_bound_gb = required_static_memory_gb * (1 - assert_static_memory_ratio)
            if lower_bound_gb < memory_epsilon_gb:
                lower_bound_gb = 0.0
            upper_bound_gb = max(
                required_static_memory_gb * (1 + assert_static_memory_ratio),
                memory_epsilon_gb
            )
            actual_gb = static_ram_usage_bytes / 1000 ** 3
            if actual_gb < lower_bound_gb or actual_gb > upper_bound_gb:
                failures.append(
                    f"Static RAM usage varies from the required static memory by more than {assert_static_memory_ratio * 100}%. " +
                    f"Expected {human_size(required_static_memory_gb * 1000 ** 3)}, got {human_size(static_ram_usage_bytes)}."
                )

        if assert_runtime_memory_ratio is not None and cls.measure_memory:
            if required_runtime_memory_gb is None:
                required_runtime_memory_gb = 0
            lower_bound_gb = required_runtime_memory_gb * (1 - assert_runtime_memory_ratio)
            if lower_bound_gb < memory_epsilon_gb:
                lower_bound_gb = 0.0
            upper_bound_gb = max(
                required_runtime_memory_gb * (1 + assert_runtime_memory_ratio),
                memory_epsilon_gb
            )
            actual_gb = runtime_ram_usage_bytes / 1000 ** 3
            if actual_gb < lower_bound_gb or actual_gb > upper_bound_gb:
                failures.append(
                    f"Runtime RAM usage varies from the required runtime memory by more than {assert_runtime_memory_ratio * 100}%. " +
                    f"Expected {human_size(required_runtime_memory_gb * 1000 ** 3)}, got {human_size(runtime_ram_usage_bytes)}."
                )

        if requires_gpu and cls.measure_gpu_memory:
            if assert_static_memory_ratio is not None:
                if required_static_gpu_memory_gb is None:
                    required_static_gpu_memory_gb = 0
                lower_bound_gb = required_static_gpu_memory_gb * (1 - assert_static_memory_ratio)
                if lower_bound_gb < memory_epsilon_gb:
                    lower_bound_gb = 0.0
                upper_bound_gb = max(
                    required_static_gpu_memory_gb * (1 + assert_static_memory_ratio),
                    memory_epsilon_gb
                )
                actual_gb = static_vram_usage_bytes / 1000 ** 3
                if actual_gb < lower_bound_gb or actual_gb > upper_bound_gb:
                    failures.append(
                        f"Static VRAM usage varies from the required static GPU memory by more than {assert_static_memory_ratio * 100}%. " +
                        f"Expected {human_size(required_static_gpu_memory_gb * 1000 ** 3)}, got {human_size(static_vram_usage_bytes)}."
                    )

            if assert_runtime_memory_ratio is not None:
                if required_runtime_gpu_memory_gb is None:
                    required_runtime_gpu_memory_gb = 0
                lower_bound_gb = required_runtime_gpu_memory_gb * (1 - assert_runtime_memory_ratio)
                if lower_bound_gb < memory_epsilon_gb:
                    lower_bound_gb = 0.0
                upper_bound_gb = max(
                    required_runtime_gpu_memory_gb * (1 + assert_runtime_memory_ratio),
                    memory_epsilon_gb
                )
                actual_gb = runtime_vram_usage_bytes / 1000 ** 3
                if actual_gb < lower_bound_gb or actual_gb > upper_bound_gb:
                    failures.append(
                        f"Runtime VRAM usage varies from the required runtime GPU memory by more than {assert_runtime_memory_ratio * 100}%. " +
                        f"Expected {human_size(required_runtime_gpu_memory_gb * 1000 ** 3)}, got {human_size(runtime_vram_usage_bytes)}."
                    )
        if len(failures) > 0:
            raise RuntimeError(f"Memory assertion(s) failed. {' '.join(failures)}")
        return result

    """Utility methods"""

    def get_model_file(self, url: str, assert_exists: bool=True) -> str:
        """
        Gets a model file from a directory.
        """
        file_name = get_file_name_from_url(url)
        file_path = os.path.join(self.model_dir, file_name)
        if assert_exists and not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        return file_path

    def reset_steps(self) -> None:
        """
        Reset the step counter.
        """
        self.num_steps = 1

    def increment_step(self) -> None:
        """
        Increment the step.
        """
        self.step += 1

    def add_intermediate(self, intermediate: Any) -> None:
        """
        Add an intermediate result.
        """
        self._intermediates.append(intermediate)

    def get_output_from_audio_result(
        self,
        result: Union[torch.Tensor, np.ndarray[Any, Any]],
        output_format: Optional[AUDIO_OUTPUT_FORMAT_LITERAL]="wav",
        sample_rate: int=DEFAULT_SAMPLE_RATE,
        normalization_strategy: Optional[Literal["clip", "peak", "rms", "loudness", "none"]]="loudness",
        strip_silence: bool=True,
        output_upload: bool=False,
        return_first_item: bool=False
    ) -> AudioResultType:
        """
        Get the output from the result based on the output type requested.
        Should not be called if the output of the model is already a file.
        """
        if strip_silence:
            result = trim_silence(result, raise_when_all_silent=False)

        if output_format is None or output_format in ["wav", "mp3", "ogg", "flac"]:
            # Save each audio
            result_uris: List[str] = []
            result_bytes: List[EncodedAudioProxy] = []
            for audio in result:
                if is_numpy_array(audio):
                    audio = torch.from_numpy(audio)
                if len(audio.shape) == 1:
                    audio = audio.unsqueeze(0)
                num_channels, num_samples = audio.shape
                duration = num_samples / sample_rate
                output_file = tempfile.mktemp()
                output_file = str(
                    audio_write(
                        output_file,
                        audio,
                        sample_rate=sample_rate,
                        format="wav" if output_format is None else output_format,
                        normalize=normalization_strategy is not None,
                        strategy=normalization_strategy,
                    )
                )
                if output_upload:
                    result_uris.append(output_file)
                else:
                    with open(output_file, "rb") as f:
                        result_bytes.append(
                            EncodedAudioProxy(
                                data=f.read(),
                                format="wav" if output_format is None else output_format,
                                sample_rate=sample_rate,
                                channels=num_channels,
                                duration=duration,
                            )
                        )
            if output_upload:
                result = result_uris # type: ignore[assignment]
            else:
                result = result_bytes # type: ignore[assignment]
        elif output_format == "float":
            # Float tensor
            if is_numpy_array(result):
                import torch
                result = torch.from_numpy(result)
            if len(result.shape) == 1:
                result = result.unsqueeze(0) # type: ignore[union-attr]
            if len(result.shape) == 2:
                result = result.unsqueeze(0) # type: ignore[union-attr]
        elif output_format == "int":
            # NP int16 array
            if is_torch_tensor(result):
                result = result.numpy()
            if len(result.shape) == 1:
                result = result[None, :] # Add channel dimension
            if len(result.shape) == 2:
                result = result[None, :] # Add batch dimension
        else:
            raise ValueError(f"Output type {output_format} not recognized.")

        if output_upload:
            result = self.save_audio_output( # type: ignore[assignment]
                result,
                output_format="wav" if output_format is None else output_format
            )

        if return_first_item:
            return result[0]
        return result

    def get_output_from_image_result(
        self,
        result: ImageType,
        output_format: Optional[IMAGE_OUTPUT_FORMAT_LITERAL]="png",
        output_upload: bool=False,
        return_first_item: bool=False
    ) -> ImageType:
        """
        Get the output from the result based on the output type requested.
        """
        if output_format is None or output_format == "png":
            # We use a simple PIL conversion here
            result = to_pil_array(result)
        elif output_format == "jpeg":
            # We convert to numpy then encode with TurboJPEG
            result = to_jpeg_array(result)
        elif output_format in ["float", "latent"]:
            # We convert to a float tensor
            # 'latent' only means something to models that operate on latent space
            result = to_bchw_tensor(result)
        elif output_format == "int":
            # We convert to a numpy uint8 array
            result = to_bhwc_ndarray(result)
        else:
            raise ValueError(f"Output type {output_format} not recognized.")
        if output_upload:
            result = self.save_image_output(
                result,
                output_format="png" if output_format is None else output_format
            )
        if return_first_item:
            return result[0]
        return result

    def get_output_from_video_result(
        self,
        result: ImageType,
        audio: Optional[AudioType]=None,
        audio_sample_rate: int=DEFAULT_SAMPLE_RATE,
        frame_rate: int=DEFAULT_FRAME_RATE,
        output_format: Optional[VIDEO_OUTPUT_FORMAT_LITERAL]="mp4",
        output_upload: bool=False,
        multi_video: bool=False
    ) -> ImageType:
        """
        Get the output from the result based on the output type requested.
        """
        num_videos = 1 if not multi_video else len(result) # type: ignore[arg-type]
        if output_format is None or output_format == "mp4":
            result_uris: List[str] = [tempfile.mktemp(suffix=".mp4") for _ in range(num_videos)]
            result_bytes: List[EncodedVideoProxy] = []
            for i in range(num_videos):
                Video(
                    frames=to_pil_array(result[i]), # type: ignore[index]
                    frame_rate=frame_rate,
                    audio=audio, # type: ignore[arg-type] # TODO: support all audio types
                    audio_rate=audio_sample_rate,
                ).save(result_uris[i])

                if not output_upload:
                    with open(result_uris[i], "rb") as f:
                        result_bytes.append(
                            EncodedVideoProxy(
                                data=f.read(),
                                format="mp4" if output_format is None else output_format,
                                frame_rate=frame_rate,
                                audio_rate=audio_sample_rate,
                            )
                        )

            if output_upload:
                result = result_uris
            else:
                result = result_bytes # type: ignore[assignment]
        elif output_format == "gif":
            gif_result_uris: List[str] = [tempfile.mktemp(suffix=".gif") for _ in range(num_videos)]
            gif_result_bytes: List[EncodedImageProxy] = []
            for i in range(num_videos):
                Video(
                    frames=to_pil_array(result[i]), # type: ignore[index]
                    frame_rate=frame_rate,
                ).save(gif_result_uris[i])

                if not output_upload:
                    with open(gif_result_uris[i], "rb") as f:
                        gif_result_bytes.append(
                            EncodedImageProxy(
                                data=f.read(),
                                format="gif"
                            )
                        )

            if output_upload:
                result = gif_result_uris
            else:
                result = gif_result_bytes
        elif output_format == "png":
            # We use a simple PIL conversion here
            result = [ # type: ignore[assignment]
                to_pil_array(result[i]) # type: ignore[index]
                for i in range(num_videos)
            ]
        elif output_format in ["float", "latent"]:
            # We convert to a float tensor
            # 'latent' only means something to models that operate on latent space
            result = [
                to_bchw_tensor(result[i]) # type: ignore[index]
                for i in range(num_videos)
            ]
        elif output_format == "int":
            # We convert to a numpy uint8 array
            result = [
                to_bhwc_ndarray(result[i]) # type: ignore[index]
                for i in range(num_videos)
            ]
        else:
            raise ValueError(f"Output type {output_format} not recognized.")

        if output_upload:
            result = self.save_video_output(
                result,
                output_format="mp4" if output_format is None else output_format
            )
        if not multi_video:
            return result[0] # type: ignore[index]
        return result

    def save_image_output(
        self,
        images: ImageType,
        output_format: IMAGE_OUTPUT_FORMAT_LITERAL="png",
    ) -> List[str]:
        """
        Saves one or more images to the configured directory.
        """
        image_filenames: List[str] = []
        if output_format in ["png", "jpeg"]:
            for image in images: # type: ignore[union-attr]
                image_filename = f"{generate_id()}.{output_format}"
                save_path = os.path.join(self.save_dir, image_filename)
                image.save(save_path) # type: ignore[union-attr]
                image_filenames.append(image_filename)
        elif output_format in ["float", "latent"]:
            for image in images: # type: ignore[union-attr]
                image_filename = f"{generate_id()}.pt"
                save_path = os.path.join(self.save_dir, image_filename)
                import torch
                torch.save(image, save_path)
                image_filenames.append(image_filename)
        elif output_format == "int":
            for image in images: # type: ignore[union-attr]
                image_filename = f"{generate_id()}.npy"
                save_path = os.path.join(self.save_dir, image_filename)
                import numpy as np
                np.save(save_path, image) # type: ignore[arg-type]
                image_filenames.append(image_filename)
        else:
            raise ValueError(f"Format {output_format} not recognized.")
        return image_filenames

    def save_audio_output(
        self,
        audios: AudioType,
        output_format: AUDIO_OUTPUT_FORMAT_LITERAL="wav",
    ) -> List[str]:
        """
        Saves one or more audios to the configured directory.
        """
        audio_filenames: List[str] = []
        if output_format in ["wav", "mp3", "ogg", "flac"]:
            for audio in audios: # type: ignore[union-attr]
                assert isinstance(audio, str), "Audio output should have been saved to a temporary file."
                audio_filename = f"{generate_id()}.{output_format}"
                audio_filenames.append(audio_filename)
                output_file = os.path.join(self.save_dir, audio_filename)
                os.rename(audio, output_file)
        elif output_format == "float":
            for audio in audios: # type: ignore[union-attr]
                audio_filename = f"{generate_id()}.pt"
                save_path = os.path.join(self.save_dir, audio_filename)
                import torch
                torch.save(audio, save_path)
                audio_filenames.append(audio_filename)
        elif output_format == "int":
            for audio in audios: # type: ignore[union-attr]
                audio_filename = f"{generate_id()}.npy"
                save_path = os.path.join(self.save_dir, audio_filename)
                import numpy as np
                np.save(save_path, audio)
                audio_filenames.append(audio_filename)
        else:
            raise ValueError(f"Format {output_format} not recognized.")
        return audio_filenames

    def save_video_output(
        self,
        videos: ImageType,
        output_format: VIDEO_OUTPUT_FORMAT_LITERAL="mp4",
    ) -> List[str]:
        """
        Saves one or more audios to the configured directory.
        """
        video_filenames: List[str] = []
        if output_format in ["mp4", "gif"]:
            for video in videos: # type: ignore[union-attr]
                assert isinstance(video, str), "Audio output should have been saved to a temporary file."
                video_filename = f"{generate_id()}.{output_format}"
                video_filenames.append(video_filename)
                output_file = os.path.join(self.save_dir, video_filename)
                os.rename(video, output_file)
        elif output_format == "png":
            from PIL import Image
            for video in videos: # type: ignore[union-attr]
                assert isinstance(video, list), "Video output should have been converted to a list of PIL images."
                video_id = generate_id()
                num_frames = len(video)
                num_frame_digits = len(str(num_frames))
                for i, frame in enumerate(video):
                    assert isinstance(frame, Image.Image), "Video output should have been converted to PIL images."
                    frame_filename = f"{video_id}_{str(i).zfill(num_frame_digits)}.png"
                    frame_path = os.path.join(self.save_dir, frame_filename)
                    frame.save(frame_path)
                    video_filenames.append(frame_filename)                   
        elif output_format == "float":
            for video in videos: # type: ignore[union-attr]
                video_filename = f"{generate_id()}.pt"
                save_path = os.path.join(self.save_dir, video_filename)
                import torch
                torch.save(video, save_path)
                video_filenames.append(video_filename)
        elif output_format == "int":
            for video in videos: # type: ignore[union-attr]
                video_filename = f"{generate_id()}.npy"
                save_path = os.path.join(self.save_dir, video_filename)
                import numpy as np
                np.save(save_path, video) # type: ignore[arg-type]
                video_filenames.append(video_filename)
        else:
            raise ValueError(f"Format {output_format} not recognized.")
        return video_filenames

    def get_option(self, option_name: str, default: Any=NOTSET) -> Any:
        """
        Get an option from the configuration.
        """
        value = self.options.get(option_name, default)
        if value is NOTSET:
            raise ValueError(f"Option {option_name} not found in configuration.")
        return value

    def set_option(self, option_name: str, value: Any) -> None:
        """
        Set an option in the configuration.
        """
        self.config[option_name] = value

    """Public methods"""

    def __hash__(self) -> int:
        """
        Hash the task and model.
        """
        return hash(self.task) + hash(self.model)

    """Overridable methods"""

    def get_offload_models(self) -> Union[str, List[str], bool]:
        """
        Get the offload models.
        """
        return False if self.offload_models is None else self.offload_models

    def get_offload_tasks(self) -> Union[str, List[str], bool]:
        """
        Get the offload tasks.
        """
        return False if self.offload_tasks is None else self.offload_tasks

    def load(self) -> None:
        """
        Load the task.
        Default triggers the pretrained loader.
        """
        # Models
        loadable_models = list(({} if not self.pretrained_models else self.pretrained_models).keys())
        offload_models = self.get_offload_models()
        offload_model_names: List[str] = []

        if offload_models == True:
            offload_model_names = loadable_models
        elif isinstance(offload_models, list):
            offload_model_names = offload_models

        self.pretrained.load(
            loadable_names=loadable_models,
            offload_names=offload_model_names,
            use_tqdm=self.use_tqdm
        )

        # Tasks
        loadable_tasks = list(({} if not self.component_tasks else self.component_tasks).keys())
        offload_tasks = self.get_offload_tasks()
        offload_task_names: List[str] = []

        if offload_tasks == True:
            offload_task_names = loadable_tasks
        elif isinstance(offload_tasks, list):
            offload_task_names = offload_tasks

        self.tasks.load(
            loadable_names=loadable_tasks,
            offload_names=offload_task_names,
            use_tqdm=self.use_tqdm
        )

    def compile(self) -> None:
        """
        Compile the task.
        Default triggers the pretrained loader.
        """
        self.pretrained.compile(use_tqdm=self.use_tqdm)
        self.tasks.compile(use_tqdm=self.use_tqdm)

    def unload(self) -> None:
        """
        Unload the task.
        Default triggers the pretrained loader.
        """
        self.pretrained.unload()
        self.tasks.unload()

    def offload(self) -> None:
        """
        Offload the task (from GPU to CPU).
        Default triggers the pretrained loader.
        """
        self.pretrained.offload()
        self.tasks.offload()

    def onload(self) -> None:
        """
        Onload the task (from CPU to GPU).
        Default triggers the pretrained loader.
        """
        self.pretrained.onload()
        self.tasks.onload()

    def __call__(self, **kwargs: Any) -> Any:
        """
        Execute the task.
        Base task does nothing.
        """
        return
