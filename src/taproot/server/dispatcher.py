from __future__ import annotations

import os
import logging
import asyncio

from typing import (
    Any,
    Coroutine,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
    Tuple,
    Type,
    Union,
    cast,
)
from typing_extensions import Literal, Self
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter

from ..util import (
    asyncio_run,
    logger,
    find_free_memory_port,
    find_free_unix_socket,
    find_free_port,
    generate_id
)

from ..tasks import Task, ExecutableTask, ExecutableTaskModel

from ..constants import *
from ..payload import *
from ..config import *

from .config import ConfigServer, ConfigType
from .pool import TrackingProcessPoolExecutor, TrackingThreadPoolExecutor

if TYPE_CHECKING:
    from .executor import Executor

@dataclass
class RunningExecutor:
    """
    Represents a running executor.
    """
    server: Executor
    future: asyncio.Future[Any]
    start_time: float = 0.0
    assignment_times: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.start_time = perf_counter()

    def recently_started(self, threshold: float=0.5) -> bool:
        """
        Check if the executor has recently started.
        """
        return perf_counter() - self.start_time < threshold

    def prune_assignment_times(self, now: float, threshold: float=1.0) -> None:
        """
        Prune assignment times.
        """
        i = 0
        while i < len(self.assignment_times) and self.assignment_times[i] < now - threshold:
            i += 1
        self.assignment_times = self.assignment_times[i:]

    def adjust_capacity(self, reported_capacity: int, threshold: float=0.5) -> int:
        """
        Adjust the reported capacity based on the assignment times.
        """
        now = perf_counter()
        self.prune_assignment_times(now)
        if not self.assignment_times:
            return reported_capacity
        assigned_in_window = 0
        for assignment_time in self.assignment_times:
            if now - assignment_time < threshold:
                assigned_in_window += 1
        if self.start_time > now - threshold:
            assigned_in_window += 1
        return max(0, reported_capacity - assigned_in_window)

    def assign(self) -> Self:
        """
        Adds an assignment time and returns the executor.
        """
        self.assignment_times.append(perf_counter())
        return self

class Dispatcher(ConfigServer):
    """
    Listens for incoming requests from the overseer, 
    and dispatches them to the appropriate executor.
    """
    config_class = DispatcherConfig
    executors: Dict[Type[Task], List[RunningExecutor]]
    pool: Optional[Union[TrackingProcessPoolExecutor, TrackingThreadPoolExecutor]]
    overseers: Set[str]

    def __init__(self, config: ConfigType = None) -> None:
        """
        Initialize the dispatcher.
        """
        super().__init__(config=config)
        self.executors = {}
        self.overseers = set()
        self.pool = None

    """Default properties"""

    @property
    def default_max_workers(self) -> int:
        """
        Default maximum number of workers for the dispatcher.
        """
        max_workers = self.config.max_workers
        if not max_workers:
            return 1
        return int(max_workers)

    @property
    def default_use_multiprocessing(self) -> bool:
        """
        Default use of multiprocessing for the dispatcher.
        """
        return bool(self.config.use_multiprocessing)

    @property
    def default_spawn_interval(self) -> Optional[float]:
        """
        Default interval between spawning new executors.
        """
        spawn_interval = self.config.spawn_interval
        if spawn_interval is None:
            return None
        return float(spawn_interval)

    """Getters/Setters"""

    @property
    def max_workers(self) -> int:
        """
        Maximum number of workers for the dispatcher.
        """
        if not hasattr(self, "_max_workers"):
            self._max_workers = self.default_max_workers
        return max(self._max_workers, self.num_preconfigured_workers)

    @max_workers.setter
    def max_workers(self, value: int) -> None:
        """
        Set the maximum number of workers for the dispatcher.
        """
        self._max_workers = value

    @property
    def spawn_interval(self) -> float:
        """
        Interval between spawning new executors.
        """
        if not hasattr(self, "_spawn_interval"):
            self._spawn_interval = self.default_spawn_interval
        return 0.0 if self._spawn_interval is None else self._spawn_interval

    @spawn_interval.setter
    def spawn_interval(self, value: Optional[float]) -> None:
        """
        Set the interval between spawning new executors.
        """
        self._spawn_interval = value

    @property
    def use_multiprocessing(self) -> bool:
        """
        Use multiprocessing for the dispatcher.
        """
        if not hasattr(self, "_use_multiprocessing"):
            self._use_multiprocessing = self.default_use_multiprocessing
        return self._use_multiprocessing

    @use_multiprocessing.setter
    def use_multiprocessing(self, value: bool) -> None:
        """
        Set the use of multiprocessing for the dispatcher.
        """
        self._use_multiprocessing = value

    @property
    def model_dir(self) -> str:
        """
        Model directory for the dispatcher.
        """
        configured = self.config.executor_config.queue_config.task_config.model_dir
        if configured is not None:
            return str(configured)
        return DEFAULT_MODEL_DIR

    @model_dir.setter
    def model_dir(self, value: str) -> None:
        """
        Set the model directory for the dispatcher.
        """
        self.config.executor_config.queue_config.task_config.model_dir = value

    @property
    def save_dir(self) -> str:
        """
        Save directory for the dispatcher.
        """
        configured = self.config.executor_config.queue_config.task_config.save_dir
        if configured is not None:
            return str(configured)
        return self.default_save_dir

    @save_dir.setter
    def save_dir(self, value: Optional[str]) -> None:
        """
        Set the save directory for the dispatcher.
        """
        self.config.executor_config.queue_config.task_config.save_dir = value

    @property
    def save_format(self) -> str:
        """
        Save format for the dispatcher.
        """
        configured = self.config.executor_config.queue_config.task_config.save_format
        if configured is not None:
            return str(configured)
        return "png"

    @save_format.setter
    def save_format(self, value: str) -> None:
        """
        Set the save format for the dispatcher.
        """
        self.config.executor_config.queue_config.task_config.save_format = value

    @property
    def executor_protocol(self) -> Literal["memory", "tcp", "ws", "unix"]:
        """
        Executor protocol for the dispatcher.
        """
        configured = self.config.executor_config.protocol
        if configured is not None:
            return str(configured) # type: ignore[return-value]
        return DEFAULT_PROTOCOL # type: ignore[return-value]

    @executor_protocol.setter
    def executor_protocol(self, value: str) -> None:
        """
        Set the executor protocol for the dispatcher.
        """
        self.config.executor_config.protocol = value

    @property
    def executor_queue_size(self) -> int:
        """
        Executor queue size for the dispatcher.
        """
        configured = self.config.executor_config.queue_config.size
        if configured is not None:
            return int(configured)
        return 1

    @executor_queue_size.setter
    def executor_queue_size(self, value: int) -> None:
        """
        Set the executor queue size for the dispatcher.
        """
        self.config.executor_config.queue_config.size = value

    """Getters only"""

    @property
    def default_save_dir(self) -> str:
        """
        Default save directory for the dispatcher.
        """
        if not hasattr(self, "_default_save_dir"):
            self._default_save_dir = os.path.join(os.getcwd(), "images")
            os.makedirs(self._default_save_dir, exist_ok=True)
        return self._default_save_dir

    @property
    def task_model_dirs(self) -> Dict[str, str]:
        """
        Model directories for each task.
        """
        if self.config.task_config is None:
            return {}
        return {
            task_key: queue.task_config.model_dir
            for task_key, queue in self.config.task_config.items()
        }

    @property
    def catalog(self) -> Dict[str, ExecutableTask]:
        """
        Catalog of tasks that can be executed by the server.
        """
        if not hasattr(self, "_catalog"):
            self._catalog = Task.catalog(
                model_dir=self.model_dir,
                task_model_dirs=self.task_model_dirs
            )
            if self.config.task_allow_list is not None:
                tasks_to_remove: List[str] = []
                for task_name, executable in self._catalog.items():
                    if task_name in self.config.task_allow_list:
                        continue
                    to_remove: List[Optional[str]] = []
                    for model_name, model in executable["models"].items():
                        if model["task"].get_key() not in self.config.task_allow_list:
                            to_remove.append(model_name)
                    for model_name in to_remove:
                        del executable["models"][model_name]
                    if not executable["models"]:
                        tasks_to_remove.append(task_name)
                self._catalog = {
                    task: executable
                    for task, executable in self._catalog.items()
                    if task not in tasks_to_remove
                }
            if self.config.task_denylist is not None:
                tasks_to_remove: List[str] = [] # type: ignore[no-redef]
                for task_name, executable in self._catalog.items():
                    if task_name not in self.config.task_denylist:
                        continue
                    to_remove: List[Optional[str]] = [] # type: ignore[no-redef] 
                    for model_name, model in executable["models"].items():
                        if model["task"].get_key() in self.config.task_denylist:
                            to_remove.append(model_name)
                    for model_name in to_remove:
                        del executable["models"][model_name]
                    if not executable["models"]:
                        tasks_to_remove.append(task_name)
                self._catalog = {
                    task: executable
                    for task, executable in self._catalog.items()
                    if task not in tasks_to_remove
                }
        return self._catalog

    @property
    def num_active_workers(self) -> int:
        """
        Number of active workers for the dispatcher.
        """
        return sum([
            len(executors)
            for executors in self.executors.values()
        ])

    @property
    def num_preconfigured_workers(self) -> int:
        """
        Number of preconfigured workers for the dispatcher.
        """
        return len(self.config.static_executor_config)

    """Private Methods"""

    def _get_task_pending_installation(
        self,
        task: str,
        model: Optional[str] = None,
    ) -> ExecutableTaskModel:
        """
        Installs a task into the catalog.
        """
        task_cls = Task.get(task, model=model, available_only=False)
        if task_cls is None:
            task_str = task
            if model is not None:
                task_str = f"{task}:{model}"
            raise ValueError(f"Could not find task '{task_str}' to install.")

        task_key = task_cls.get_key()
        if self.config.task_allow_list is not None and task not in self.config.task_allow_list and task_key not in self.config.task_allow_list:
            raise ValueError(f"Task '{task_key}' is not in the allow_list.")
        if self.config.task_denylist is not None and (task in self.config.task_denylist or task_key in self.config.task_denylist):
            raise ValueError(f"Task '{task_key}' is in the denylist.")

        if task not in self.catalog:
            self.catalog[task] = {
                "default": model if task_cls.default else None,
                "models": {
                    model: {
                        "task": task_cls,
                        **task_cls.introspect()
                    }
                }
            }
        else:
            self.catalog[task]["models"][model] = {
                "task": task_cls,
                **task_cls.introspect()
            }

        return self.catalog[task]["models"][model]

    def _get_executable_for_task(
        self,
        task: str,
        model: Optional[str] = None,
        install: bool = False
    ) -> ExecutableTaskModel:
        """
        Get an executable task from the catalog.
        """
        if task not in self.catalog:
            if install:
                return self._get_task_pending_installation(task, model) # Will raise
            raise ValueError(f"Task '{task}' not found in catalog. Valid tasks are: {list(self.catalog.keys())}")
        if model is None:
            model = self.catalog[task]["default"]
            if model is None and None not in self.catalog[task]["models"]:
                model = list(self.catalog[task]["models"].keys())[0]
        if model not in self.catalog[task]["models"]:
            if install:
                return self._get_task_pending_installation(task, model) # Will raise
            raise ValueError(f"Model '{model}' not found in task '{task}' catalog. Valid models are: {list(self.catalog[task]['models'].keys())}")
        return self.catalog[task]["models"][model]

    def _get_executor_config_for_task(
        self,
        task: Type[Task],
        capability: CapabilityPayload,
        **parameters: ParameterMetadataPayload
    ) -> ExecutorConfigDict:
        """
        Get the executor configuration for a given task.
        """
        if self.config.task_config is not None:
            task_key = task.get_key()
            if task_key in self.config.task_config:
                queue_config = self.config.task_config[task_key]
            elif task.task in self.config.task_config:
                queue_config = self.config.task_config[task.task]
            else:
                queue_config = self.config.executor_config.queue_config
        else:
            queue_config = self.config.executor_config.queue_config

        gpu_index = queue_config.task_config.gpu_index

        if gpu_index is None:
            gpu_index = self.get_capability().get_optimal_gpu_id(
                **self._get_capability_requirements_for_task(
                    task,
                    capability,
                    spawning=True,
                    **parameters
                )
            )

        protocol = self.config.executor_config.protocol
        config_dict: ExecutorConfigDict = {
            "queue_config": {
                "task": task.task,
                "model": task.model,
                "size": queue_config.size,
                "task_config": {
                    "gpu_index": gpu_index,
                    "model_dir": queue_config.task_config.model_dir,
                    "dtype": queue_config.task_config.dtype,
                    "save_dir": queue_config.task_config.save_dir or self.save_dir,
                }
            },
            "protocol": protocol,
            "host": None,
            "port": None,
            "encryption": None,
            "allow_list": self.config.executor_config.allow_list,
            "max_idle_time": self.config.executor_config.max_idle_time,
        }

        if protocol == "memory":
            config_dict["port"] = find_free_memory_port()
        elif protocol in ["tcp", "ws"]:
            if self.config.executor_config.host:
                config_dict["host"] = self.config.executor_config.host
            elif self.protocol != "unix":
                config_dict["host"] = self.host
            config_dict["port"] = find_free_port()
            if self.config.executor_config.encryption:
                config_dict["encryption"] = cast(
                    EncryptionConfigDict,
                    dict(self.config.executor_config.encryption)
                )
        elif protocol == "unix":
            config_dict["path"] = find_free_unix_socket()

        if self.use_control_encryption:
            config_dict["control_encryption"] = {
                "encryption_key": self.control_encryption_key,
                "encryption_use_aesni": self.control_encryption_use_aesni
            }

        return config_dict

    def _prune_executors_for_task(self, task: Type[Task]) -> None:
        """
        Trim the cache for a given task.
        """
        if task in self.executors:
            still_running: List[RunningExecutor] = []
            for executor in self.executors[task]:
                if not executor.future.done():
                    still_running.append(executor)
            self.executors[task] = still_running

    def _can_spawn_executor_for_task(self, task: Type[Task]) -> bool:
        """
        Check if a new executor can be spawned for a given task.
        """
        # Check if there is one currently spawning
        check_time = perf_counter()
        for executor in self.executors.get(task, []):
            if executor.future.done():
                continue
            if check_time - executor.start_time < self.spawn_interval:
                return False
        if self.max_workers <= self.num_active_workers:
            return False
        if self.config.task_max_workers is None:
            return True
        task_key = task.get_key()
        if task_key not in self.config.task_max_workers:
            return True
        num_workers = self.config.task_max_workers[task_key]
        return bool(len(self.executors.get(task, [])) < num_workers)

    async def _maybe_prune_idle_executors(self, to_prune: int=1) -> bool:
        """
        Prune any idle executors to free up resources.
        """
        # Run a prune first, then get running executors
        running_executors: List[Tuple[Type[Task], RunningExecutor]] = []
        num_running_executors: Dict[str, int] = {}
        for task in list(self.executors.keys()):
            self._prune_executors_for_task(task)
            for executor in self.executors.get(task, []):
                running_executors.append((task, executor))
                num_running_executors[task.get_key()] = num_running_executors.get(task.get_key(), 0) + 1

        # Query statuses of running executors
        statuses = await asyncio.gather(*[
            executor.server.get_status()
            for _, executor in running_executors
        ])

        # Filter to the idle executors 
        idle_executors = [
            (task, executor, status)
            for (task, executor), status in zip(running_executors, statuses)
            if status.get("status", "active") == "idle"
            and executor.server.allocation != "static"
            and not executor.recently_started()
        ]

        # Sort by activity
        idle_executors.sort(key=lambda x: float(x[2].get("activity", 0.0))) # type: ignore[arg-type]

        if len(idle_executors) < to_prune:
            return False

        # Send graceful exit to the idle executors
        await asyncio.gather(*[
            executor.server.exit()
            for _, executor, _ in idle_executors[:to_prune]
        ])

        # Prune the executors again
        pruned_tasks = set()
        for task, executor, _ in idle_executors[:to_prune]:
            if task not in pruned_tasks:
                self._prune_executors_for_task(task)
                pruned_tasks.add(task)

        return True

    async def _get_executor_and_status_for_task(
        self,
        task: Type[Task],
        timeout: Optional[float]=0.1,
        retries: int=0,
        request_id: Optional[str]=None,
        client_id: Optional[str]=None
    ) -> List[Tuple[RunningExecutor, Dict[str, Any]]]:
        """
        Get an available executor for a given task.
        """
        executors = self.executors.get(task, [])
        data: Optional[str] = None
        if request_id and client_id:
            data = f"{client_id}:{request_id}"
        statuses = await asyncio.gather(*[
            executor.server.get_status(
                data=data,
                timeout=0.2,
                retries=retries,
            )
            for executor in executors
        ])
        return list(zip(executors, statuses)) # type: ignore[arg-type]

    async def _get_available_executor_for_task(
        self,
        task: Type[Task],
        timeout: Optional[float]=0.1,
        retries: int=0,
        request_id: Optional[str]=None,
        client_id: Optional[str]=None
    ) -> Optional[RunningExecutor]:
        """
        Get an available executor for a given task.
        """
        idle_executor: Optional[RunningExecutor] = None
        executor_status = await self._get_executor_and_status_for_task(
            task,
            timeout=timeout,
            retries=retries,
            request_id=request_id,
            client_id=client_id
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Got {len(executor_status)} executors for task {task.get_key()}")
            for executor, status in executor_status:
                logger.debug(f"Executor {executor.server.address} status: {status}")
        for i, (executor, status) in enumerate(executor_status):
            if status.get("has_id", False):
                logger.debug(f"Returning executor {executor.server.address} with ID for task {task.get_key()}")
                return executor # Return immediately if the ID is found
            elif status.get("status", "active") == "idle" and not executor.recently_started():
                idle_executor = executor
                break
        if idle_executor:
            logger.debug(f"Returning idle executor {idle_executor.server.address} for task {task.get_key()}")
            return idle_executor.assign()
        # If no idle executor, look for ones that have queue capacity
        for i, (executor, status) in enumerate(executor_status):
            reported_capacity = status.get("capacity", 0)
            adjusted_capacity = executor.adjust_capacity(reported_capacity)
            if adjusted_capacity > 0:
                logger.debug(f"Returning executor {executor.server.address} with capacity {adjusted_capacity} for task {task.get_key()}")
                return executor.assign()
        logger.debug(f"No available executors for task {task.get_key()}")
        return None

    async def _spawn_executor(
        self,
        task: Type[Task],
        capability: CapabilityPayload,
        disable_idle_timeout: bool=False,
        **parameters: ParameterMetadataPayload
    ) -> RunningExecutor:
        """
        Spawn a new executor for a given task.
        """
        if getattr(self, "pool", None) is None:
            raise RuntimeError("Pool is not initialized.")

        if self.manual_exit:
            raise RuntimeError("Dispatcher is exiting, cannot spawn new executor.")

        while not self.pool.has_available_workers: # type: ignore[union-attr]
            await self._maybe_prune_idle_executors()
            await asyncio.sleep(0.01)

        from .executor import Executor
        executor_config = self._get_executor_config_for_task(task, capability, **parameters)
        if disable_idle_timeout:
            executor_config["max_idle_time"] = None

        executor = Executor(executor_config) # type: ignore[arg-type]
        future = asyncio.get_running_loop().run_in_executor(
            self.pool,
            executor.serve
        )

        if task not in self.executors:
            self.executors[task] = []

        now_running = RunningExecutor(server=executor, future=future)
        try:
            await asyncio.sleep(0.001)
            await executor.assert_connectivity()
        except Exception as e:
            logger.error(f"Error spawning executor for task {task}: {e}")
            raise e

        self.executors[task].append(now_running) # Add to memory before asserting connectivity
        return now_running.assign()


    async def _prepare_task(
        self,
        task: Type[Task],
        capability: CapabilityPayload,
        request_id: str,
        client_id: str,
        retries: int = DISPATCHER_RESERVE_MAX_RETRIES,
        **parameters: ParameterMetadataPayload
    ) -> str:
        """
        Prepare the task for execution.
        """
        # Prune the executors for the task
        self._prune_executors_for_task(task)

        # First check if an executor is available
        available_executor = await self._get_available_executor_for_task(task, client_id=client_id, request_id=request_id)

        # Now run a new executor if none are available
        if not available_executor:
            if not self._can_spawn_executor_for_task(task):
                logger.debug(f"Can't spawn new executor for task {task.get_key()} - no capacity ({self.num_active_workers}/{self.max_workers})")
                raise RuntimeError(f"Cannot spawn new executor for task {task.get_key()} - maximum number of workers reached.")
            else:
                logger.info(f"Spawning new executor for task {task}")
                available_executor = await self._spawn_executor(
                    task,
                    capability,
                    disable_idle_timeout=False,
                    **parameters
                )

        return available_executor.server.external_address

    def _get_capability_requirements_for_task(
        self,
        task: Type[Task],
        capability: CapabilityPayload,
        spawning: bool=False,
        **parameters: ParameterMetadataPayload
    ) -> Dict[str, Any]:
        """
        Get the capability requirements for a given task.
        """
        use_gpu = task.requires_gpu()

        if use_gpu:
            precision = task.required_gpu_precision()
            required_cuda_version = task.required_cuda_version()
            if precision in ["half", "float16", "fp16"]:
                max_gpu_performance = capability.get("gpu_half_float_performance_gflop_s", None)
            elif precision in ["double", "float64", "fp64"]:
                max_gpu_performance = capability.get("gpu_double_float_performance_gflop_s", None)
            else:
                max_gpu_performance = capability.get("gpu_single_float_performance_gflop_s", None)
        else:
            precision = None
            required_cuda_version = None
            max_gpu_performance = None

        required_memory_gb = 0.0
        required_gpu_memory_gb = 0.0

        if spawning:
            required_static_memory_gb = task.required_static_memory_gb()
            if required_static_memory_gb is not None:
                required_memory_gb += required_static_memory_gb
        required_runtime_memory_gb = task.required_runtime_memory_gb(**parameters)
        if required_runtime_memory_gb is not None:
            required_memory_gb += required_runtime_memory_gb

        if use_gpu:
            if spawning:
                required_static_gpu_memory_gb = task.required_static_gpu_memory_gb()
                if required_static_gpu_memory_gb is not None:
                    required_gpu_memory_gb += required_static_gpu_memory_gb
            required_runtime_gpu_memory_gb = task.required_runtime_gpu_memory_gb(**parameters)
            if required_runtime_gpu_memory_gb is not None:
                required_gpu_memory_gb += required_runtime_gpu_memory_gb

        return {
            "use_gpu": use_gpu,
            "gpu_precision": precision,
            "required_cuda_version": required_cuda_version,
            "required_memory_gb": required_memory_gb,
            "required_gpu_memory_gb": required_gpu_memory_gb,
            "max_gpu_memory_bandwidth": capability.get("gpu_memory_bandwidth_gb_s", None),
            "max_gpu_performance": max_gpu_performance,
        }

    async def _spawn_configured_executors(self) -> None:
        """
        Spawn the configured executors after the dispatcher starts.
        """
        from .executor import Executor
        for executor_config in self.config.static_executor_config:
            logger.info(f"Spawning static executor for task {executor_config.queue_config.task}")
            # Find the task in the catalog (will raise an error if not found)
            executable = self._get_executable_for_task(
                executor_config.queue_config.task,
                executor_config.queue_config.model,
                executor_config.install
            )
            task = executable["task"]
            executor_config.allocation = "static"
            if self.use_control_encryption:
                executor_config.control_encryption = EncryptionConfig()
                executor_config.control_encryption.encryption_key = self.control_encryption_key
                executor_config.control_encryption.encryption_use_aesni = self.control_encryption_use_aesni
            executor = Executor(executor_config)
            future = asyncio.get_running_loop().run_in_executor(
                self.pool,
                executor.serve
            )

            if task not in self.executors:
                self.executors[task] = []

            now_running = RunningExecutor(server=executor, future=future)
            self.executors[task].append(now_running)

    """Public Methods"""

    async def register_overseer(self, address: str) -> None:
        """
        Send a registration request to the target overseer for this dispatcher.
        We don't mind if we're already registered - the overseer can handle that.
        """
        try:
            await self.get_client_for_address(address)(
                self.pack_control_message("register", self.address),
                timeout=0.4
            )
            self.overseers.add(address)
            logger.info(f"Registered dispatcher on {self.address} with overseer on {address}")
        except Exception as e:
            logger.error(f"Error registering dispatcher on {self.address} with overseer on {address}")
            raise e

    async def unregister_overseer(self, address: str) -> None:
        """
        Send an unregistration request to the target overseer for this dispatcher.
        We don't mind if we're already unregistered - the overseer can handle that.
        """
        try:
            await self.get_client_for_address(address)(
                self.pack_control_message("unregister", self.address),
                timeout=0.4
            )
            self.overseers.remove(address)
            logger.info(f"Unregistered dispatcher on {self.address} with overseer on {address}")
        except Exception as e:
            logger.error(f"Error unregistering dispatcher on {self.address} with overseer on {address}")
            raise e

    """Overrides"""

    @contextmanager
    def context(self) -> Iterator[None]:
        """
        Context manager for the dispatcher.
        """
        with super().context():
            if self.use_multiprocessing:
                self.pool = TrackingProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self.pool = TrackingThreadPoolExecutor(max_workers=self.max_workers)
            asyncio_run(self._spawn_configured_executors())
            yield

    async def command(self, request: str, data: Any=None) -> Any:
        """
        Process a command for the dispatcher.
        """
        if request not in ["score", "prepare"]:
            return await super().command(request, data)

        # Gather details from the metadata
        assert isinstance(data, dict), "Metadata required to score a task"
        task = data.get("task", None)
        model = data.get("model", None)
        assert task is not None, "Task required to score a task"

        # Find the task in the catalog (will raise an error if not found)
        executable = self._get_executable_for_task(task, model)
        task_class = executable["task"]

        # Gather more details from the metadata
        request_id = data.get("id", None)
        client_id = data.get("client_id", None)
        cluster_capability = data.get("cluster_capability", {})
        parameters = data.get("parameters", {})

        if client_id is None:
            logger.warning("Dispatcher received request without client ID. Generating a random one instead - this may cause issues with task tracking.")
            client_id = generate_id()
        if request_id is None:
            request_id = generate_id()

        if request == "score":
            # Check if the task is available
            if not task_class.is_available():
                logger.debug(f"Task {task_class} is not available, returning 0 score.")
                return 0

            # Get executors and status for the task
            executor_statuses = await self._get_executor_and_status_for_task(
                task_class,
                request_id=request_id,
                client_id=client_id
            )

            total_executor_capacity = 0
            for executor, status in executor_statuses:
                if status.get("has_id", False):
                    return AVAILABILITY_SCORE_MAX # Return max score if the ID is found
                total_executor_capacity += status.get("capacity", 0)
            if total_executor_capacity == 0 and not self._can_spawn_executor_for_task(task_class):
                # No more queue spots and cannot spawn new executors, see if we can prune idle executors
                await self._maybe_prune_idle_executors()
                if not self._can_spawn_executor_for_task(task_class):
                    logger.debug(f"Task {task_class} has no more capacity and cannot spawn new executors, returning 0 score.")
                    return 0

            # If we're here, there's active capacity or we can spawn a new executor
            # Gather details for scoring calculation
            capability_kwargs = self._get_capability_requirements_for_task(
                task_class,
                cluster_capability,
                spawning=total_executor_capacity == 0,
                **parameters
            )
            # Score the task
            score = self.get_capability().score(**capability_kwargs)
            logger.debug(f"Task {task_class} scored {score} with capability requirements {capability_kwargs}")
            return score
        elif request == "prepare":
            logger.debug(f"Prepare request received for task {task_class} with ID {request_id} and client ID {client_id}")
            executor_address = await self._prepare_task(
                task_class,
                cluster_capability,
                request_id=request_id,
                client_id=client_id,
                **parameters
            )
            return {
                "address": executor_address,
                "id": request_id,
            }
        return None # type: ignore[unreachable,unused-ignore]

    async def status(self, data: Any=None) -> DispatcherStatusPayload:
        """
        Get the status of the dispatcher.
        """
        status = cast(DispatcherStatusPayload, await super().status(data))

        # Gather list of executors
        status["overseers"] = list(self.overseers)
        status["executors"] = {}
        to_query: List[Tuple[str, str]] = []
        for task, executors in self.executors.items():
            status["executors"][task.get_key()] = {}
            for executor in executors:
                status["executors"][task.get_key()][executor.server.address] = {} # type: ignore[typeddict-item,assignment,unused-ignore]
                to_query.append((task.get_key(), executor.server.address))

        # Query all executors in parallel
        statuses = await asyncio.gather(*[
            self.get_client_for_address(address)(
                self.pack_control_message("status"),
                timeout=0.1
            )
            for _, address in to_query
        ])

        # Assign parallel query results and return
        for (task_key, address), executor_status in zip(to_query, statuses):
            status["executors"][task_key][address] = executor_status

        return status

    async def handle(self, request: Any) -> Any:
        """
        Handles the request from the overseer.
        """
        raise NotImplementedError("Dispatcher does not perform any client-facing operations.")

    def shutdown(self) -> None:
        """
        Shutdown the dispatcher.
        """
        super().shutdown()

        # Go through executors and issue a shutdown to all in parallel
        executors_arrays = list(self.executors.values())
        self.executors = {}
        exit_futures: List[Coroutine[Any, Any, Any]] = []
        for executor_array in executors_arrays:
            for executor in executor_array:
                exit_futures.append(executor.server.exit())
        try:
            asyncio_run(
                asyncio.gather(*exit_futures),
                as_task=False
            )
        except Exception as e:
            logger.debug(f"Dispatcher ignoring error during executor exiting: {e}")

        # Now go through overseers and unregister from all in parallel
        unregister_futures: List[Coroutine[Any, Any, Any]] = []
        for overseer in list(self.overseers):
            unregister_futures.append(self.unregister_overseer(overseer))
        try:
            asyncio_run(
                asyncio.gather(*unregister_futures),
                as_task=False
            )
        except Exception as e:
            logger.debug(f"Dispatcher ignoring error during overseer unregistering: {e}")

        # Hard shut down anything that remains
        if getattr(self, "pool", None) is not None:
            self.pool.shutdown(wait=False, cancel_futures=True) # type: ignore[union-attr]
            del self.pool

        for executor_array in executors_arrays:
            for executor in executor_array:
                if not executor.future.done():
                    try:
                        executor.future.cancel()
                    except Exception as e:
                        logger.warn(f"Error cancelling task, ignoring: {e}")
                        continue
