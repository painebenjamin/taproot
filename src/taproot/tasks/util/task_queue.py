from __future__ import annotations

import asyncio
import weakref
import threading
import traceback

from typing import Any, Dict, Literal, Optional, Tuple, List, Callable, cast, TYPE_CHECKING
from typing_extensions import TypedDict

from math import exp
from time import perf_counter
from collections import deque

from taproot.config import ConfigMixin, TaskQueueConfig, TaskConfig
from taproot.util import logger, get_payload_id, execute_and_await

if TYPE_CHECKING:
    from taproot.tasks.base import Task

__all__ = ["TaskQueue", "TaskQueueResult"]

class TaskQueueResult(TypedDict):
    """
    Result of calling a task queue.
    """
    id: str # encoded hash
    status: Literal["new", "queued", "active", "complete", "error"]
    progress: float
    result: Any
    intermediate: Any
    rate: Optional[float]
    start: Optional[float]
    end: Optional[float]
    duration: Optional[float]
    remaining: Optional[float]
    callback: Optional[Any]

class TaskQueue(ConfigMixin):
    """
    A queue of tasks to be executed.
    Also maintains a list of results, stored by hashed arguments.
    Should be initialized with the task name and model.
    """
    config_class = TaskQueueConfig
    _task: Task # Taproot task
    _queue: deque[Tuple[str, Dict[str, Any]]] # queue of arguments
    _periodic_task: Optional[asyncio.Task[Any]] # periodically checks queue
    _active_task: Optional[asyncio.Task[Any]] # actively executing task
    _load_task: Optional[asyncio.Task[Any]] # loads the task
    _active_id: Optional[str] # encoded hashed arguments of active task
    _job_progress: Optional[float] # active job progress
    _job_task: Optional[str] # active job task (reported by task)
    _job_results: Dict[str, Any] # encoded hashed arguments to result
    _job_starts: Dict[str, float] # base64-encoded hashed arguments to start time
    _job_ends: Dict[str, float] # base64-encoded hashed arguments to end time
    _job_access: Dict[str, float] # base64-encoded hashed arguments to access time
    _job_callback: Dict[str, Callable[[Any], Any]] # callbacks for each job
    _job_callback_result: Dict[str, Any] # results of callbacks
    _util_ema: float # utilization exponential moving average
    _active_update: float # active task update time

    def __init__(self, config: Optional[Dict[str, Any]]=None) -> None:
        super(TaskQueue, self).__init__(config)
        self._queue = deque()
        self._job_results = {}
        self._job_starts = {}
        self._job_ends = {}
        self._job_access = {}
        self._job_callback = {}
        self._job_callback_result = {}
        self._periodic_task = None
        self._active_task = None
        self._active_id = None
        self._job_progress = None
        self._job_task = None
        self._util_ema = 0.0
        self._executions = 0
        self._lock = threading.Lock()

    """Configuration attributes"""

    @property
    def task_name(self) -> str:
        """
        Returns the name of the task.
        """
        return str(self.config.task)

    @task_name.setter
    def task_name(self, value: str) -> None:
        """
        Sets the name of the task.
        """
        self.config.task = value

    @property
    def model_name(self) -> Optional[str]:
        """
        Returns the name of the model.
        """
        if self.config.model is None:
            return None
        return str(self.config.model)

    @model_name.setter
    def model_name(self, value: Optional[str]) -> None:
        """
        Sets the name of the model.
        """
        self.config.model = value

    @property
    def result_duration(self) -> Optional[float]:
        """
        Returns the duration to keep results.
        """
        if self.config.result_duration is None:
            return None
        return float(self.config.result_duration)

    @result_duration.setter
    def result_duration(self, value: Optional[float]) -> None:
        """
        Sets the duration to keep results.
        """
        self.config.result_duration = value

    @property
    def polling_interval(self) -> float:
        """
        Returns the polling interval.
        """
        poll_interval = self.config.polling_interval
        if poll_interval is None:
            return 0.005
        return max(0.005, float(poll_interval))

    @polling_interval.setter
    def polling_interval(self, value: float) -> None:
        """
        Sets the polling interval.
        """
        self.config.polling_interval = value

    @property
    def queue_size(self) -> int:
        """
        Returns the maximum size of the queue.
        """
        configured_size = self.config.size
        if configured_size is None:
            return 1
        return max(1, int(configured_size))

    @queue_size.setter
    def queue_size(self, value: int) -> None:
        """
        Sets the maximum size of the queue.
        """
        self.config.size = value

    @property
    def task_config(self) -> TaskConfig:
        """
        Returns the task configuration.
        """
        if self.config.task_config is None:
            self.config.task_config = TaskConfig()
        return cast(TaskConfig, self.config.task_config)

    @task_config.setter
    def task_config(self, value: TaskConfig) -> None:
        """
        Sets the task configuration.
        """
        self.config.task_config = value

    @property
    def activity_tau(self) -> float:
        """
        Returns the alpha value for the active task EMA.
        """
        if self.config.activity_tau is None:
            return 30.0
        return max(1.0, float(self.config.activity_tau))

    @activity_tau.setter
    def activity_tau(self, value: float) -> None:
        """
        Sets the alpha value for the active task EMA.
        """
        self.config.activity_tau = value

    @property
    def executions(self) -> int:
        """
        Returns the number of executions.
        """
        return self._executions

    """Read-only attributes"""

    @property
    def capacity(self) -> int:
        """
        Returns the remaining capacity of the queue.
        """
        return self.queue_size - len(self._queue) + (self._active_task is None)

    @property
    def full(self) -> bool:
        """
        Checks if the queue is full.
        """
        return len(self._queue) >= self.queue_size

    @property
    def status(self) -> Literal["ready", "active", "idle", "zombie"]:
        """
        Returns the status of the queue.
        """
        if self._active_task is not None:
            return "active"
        if self.zombie:
            return "zombie"
        if self._queue:
            return "ready" # Will be started next period
        return "idle" # No queue, no active task, periodic task running

    @property
    def zombie(self) -> bool:
        """
        Checks if the queue is a zombie.
        """
        return self._periodic_task is None or self._periodic_task.done()

    @property
    def activity(self) -> float:
        """
        Returns the ratio of active time to total time.
        """
        return self._util_ema * 100.0

    @property
    def cache_length(self) -> int:
        """
        Returns the length of the cache.
        """
        return len(self._job_results)

    """Class methods"""

    @classmethod
    def get(
        cls,
        task: str,
        model: Optional[str]=None,
        **kwargs: Any
    ) -> TaskQueue:
        """
        Returns the task queue for the given task and model.
        """
        return cls(config={"task": task, "model": model, **kwargs})

    """Internal methods"""

    def _check_start_periodic_task(self) -> None:
        """
        Checks if the periodic task should be started.
        """
        if self.zombie:
            loop = asyncio.get_running_loop()
            logger.info(f"Starting periodic task for {self.task_name}:{self.model_name}")
            self._periodic_task = loop.create_task(self._periodic_check())
            weakref.finalize(self, self._check_stop_periodic_task, self._periodic_task)

    def _check_stop_periodic_task(self, *args: Any) -> None:
        """
        Called when the object is garbage collected, or explicitly during shutdown.

        Checks if the periodic task should be stopped.
        """
        if self._periodic_task is not None:
            if not self._periodic_task.done():
                logger.info(f"Stopping periodic task for {self.task_name}:{self.model_name}")
                self._periodic_task.cancel()
            self._periodic_task = None

    async def _finalize_periodic_task(self, task: asyncio.Task[Any]) -> None:
        """
        Finalizes a task by awaiting it.
        """
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _periodic_check(self) -> None:
        """
        Periodically checks the status of the queue.
        """
        try:
            while True:
                self._prune_results()
                self._check_start_next_job()
                self._update_activity()
                await asyncio.sleep(self.polling_interval)
        except asyncio.CancelledError:
            pass
        logger.debug(f"Periodic task for {self.task_name}:{self.model_name} stopped.")

    def _cleanup(self) -> None:
        """
        Cleans up the task queue.
        """
        self._job_results.clear()
        self._job_starts.clear()
        self._job_ends.clear()
        self._job_access.clear()
        self._job_callback.clear()
        self._job_callback_result.clear()
        self._active_id = None
        self._job_progress = None
        self._job_task = None

    def _start_next_job(self) -> None:
        """
        Starts the next job in the queue.
        This assumes that the existing job has already been checked for completion.
        """
        if self._queue:
            payload_id, payload = self._queue.popleft()
            self._active_id = payload_id
            loop = asyncio.get_running_loop()
            logger.debug(f"Starting job for {self.task_name}:{self.model_name} with ID {payload_id}")
            self._active_task = loop.create_task(
                self._execute_and_save_task(payload_id, **payload)
            )

    def _check_start_next_job(self) -> None:
        """
        Checks if the next job should be started.
        """
        with self._lock:
            # First check if the active task is done
            if self._active_task is not None:
                if not self._active_task.done():
                    return
                # If it is finished, clear state
                self._job_progress = None
                self._job_task = None
                self._active_task = None
                self._active_id = None

            # Now check if we should start the next job
            if self._active_task is None and self._queue:
                self._start_next_job()

    async def _initialize_task(self) -> None:
        """
        Initializes the task.
        """
        if not hasattr(self, "_task"):
            from taproot.tasks.base import Task
            try:
                task_cls = Task.get(task=self.task_name, model=self.model_name)
                if task_cls is None:
                    raise ValueError(f"Task {self.task_name}:{self.model_name} not found.")

                task_instance = task_cls(self.task_config) # type: ignore[arg-type]
                task_instance.load()
                self._task = task_instance
            except Exception as ex:
                logger.error(f"Error initializing task {self.task_name}:{self.model_name}: {type(ex).__name__} {str(ex)}")
                raise

    def _execute_task(self, payload: Dict[str, Any]) -> Any:
        """
        Executes the task with the given arguments.
        """
        try:
            return self._task(**payload)
        except Exception as e:
            logger.warning(traceback.format_exc())
            return e

    async def _execute_and_save_task(self, payload_id: str, **kwargs: Any) -> Any:
        """
        Executes the task with the given arguments and saves the result.
        """
        await self.wait_for_task()
        self._task.num_steps = 1 # Reset steps and counters
        self._job_starts[payload_id] = perf_counter()
        result = await asyncio.to_thread(self._execute_task, kwargs)

        callback = self._job_callback.pop(payload_id, None)
        callback_result: Any = None
        if callback is not None:
            callback_result = await execute_and_await(callback, result)

        with self._lock:
            end_time = perf_counter()
            self._job_results[payload_id] = result
            self._job_ends[payload_id] = end_time
            self._job_access[payload_id] = end_time
            if callback is not None:
                self._job_callback_result[payload_id] = callback_result

        return result

    def _prune_results(self) -> None:
        """
        Prunes the results dictionary.
        """
        if self.result_duration is None:
            return
        current_time = perf_counter()
        to_pop: List[str] = []
        for payload_id, result_time in self._job_access.items():
            if current_time - result_time > self.result_duration:
                to_pop.append(payload_id)
        for payload_id in to_pop:
            self._job_results.pop(payload_id, None)
            self._job_starts.pop(payload_id, None)
            self._job_ends.pop(payload_id, None)
            self._job_access.pop(payload_id, None)
            self._job_callback.pop(payload_id, None)
            self._job_callback_result.pop(payload_id, None)

    def _get_job_status(self, payload_id: str) -> Literal["new", "queued", "active", "complete", "error"]:
        """
        Returns the status of the job.
        """
        if payload_id in self._job_results:
            if isinstance(self._job_results[payload_id], Exception):
                return "error"
            return "complete"
        elif payload_id == self._active_id:
            return "active"
        elif any(p_id == payload_id for p_id, _ in self._queue):
            return "queued"
        return "new"

    def _get_job_progress(self, payload_id: str) -> float:
        """
        Returns the progress of the job.
        """
        if payload_id in self._job_results:
            return 1.0
        elif payload_id == self._active_id:
            return self._task.progress
        return 0.0

    def _get_job_rate(self, payload_id: str) -> Optional[float]:
        """
        Returns the rate of the job.
        """
        if payload_id in self._job_results:
            job_start = self._job_starts.get(payload_id, None)
            job_end = self._job_ends.get(payload_id, None)
            if job_start is not None and job_end is not None:
                elapsed_time = job_end - job_start
                if elapsed_time > 0:
                    return 1.0 / elapsed_time
        if payload_id == self._active_id:
            # Normalize the rate by the number of steps
            return self._task.rate / self._task.num_steps
        return None

    def _get_job_duration(self, payload_id: str) -> Optional[float]:
        """
        Returns the duration of the job.
        """
        job_start = self._job_starts.get(payload_id, None)
        job_end = self._job_ends.get(payload_id, None)
        if job_start is not None and job_end is not None:
            return job_end - job_start
        elif job_start is not None and payload_id == self._active_id:
            return perf_counter() - job_start
        return None

    def _get_job_remaining(self, payload_id: str) -> Optional[float]:
        """
        Returns the remaining time of the job.
        """
        if payload_id in self._job_results:
            return 0.0
        elif payload_id == self._active_id:
            return getattr(self._task, "remaining", None)
        return None

    def _get_job_intermediate(self, payload_id: str) -> Optional[Any]:
        """
        Returns the remaining time of the job.
        """
        if payload_id == self._active_id:
            return getattr(self._task, "last_intermediate", None)
        return None

    def _add_job(
        self,
        payload_id: str,
        callback: Optional[Callable[[Any], Any]]=None,
        **kwargs: Any
    ) -> None:
        """
        Adds a job to the queue.
        """
        if self.full:
            raise ValueError("Queue is full, cannot add job.")
        self._executions += 1
        self._queue.append((payload_id, kwargs))
        if callback is not None:
            self._job_callback[payload_id] = callback
        self._check_start_periodic_task() # Start the periodic task if it is not running

    def _update_activity(self) -> None:
        """
        Updates the activity of the queue.
        """
        current_time = perf_counter()
        if not hasattr(self, "_active_update"):
            self._active_update = current_time
            elapsed_time = 1e-3
        else:
            elapsed_time = current_time - self._active_update

        alpha = 1 - exp(-elapsed_time / self.activity_tau)
        is_active = self._active_task is not None or bool(self._queue)

        self._util_ema = (1 - alpha) * self._util_ema + alpha * (1.0 if is_active else 0.0)
        self._active_update = current_time

    def _unload_task(self) -> None:
        """
        Unloads the task.
        """
        with self._lock:
            if hasattr(self, "_task"):
                self._task.unload()

    def _offload_task(self) -> None:
        """
        Offloads the task (from GPU to CPU).
        """
        with self._lock:
            if hasattr(self, "_task"):
                self._task.offload()

    def _onload_task(self) -> None:
        """
        Onloads the task (from CPU to GPU).
        """
        with self._lock:
            if hasattr(self, "_task"):
                self._task.onload()

    """Public methods"""

    def start(self) -> None:
        """
        Starts the initial rounds of tasks.
        """
        loop = asyncio.get_running_loop()
        logger.info(f"Starting load task for {self.task_name}:{self.model_name}")
        self._load_task = loop.create_task(self._initialize_task())

        logger.info(f"Starting periodic task for {self.task_name}:{self.model_name}")
        self._periodic_task = loop.create_task(self._periodic_check())
        weakref.finalize(self, self._check_stop_periodic_task, self._periodic_task)

    async def wait_for_task(self, polling_interval: float=0.01) -> None:
        """
        Waits for the task to be initialized.
        """
        while not hasattr(self, "_task"):
            await asyncio.sleep(polling_interval)

    async def shutdown(self) -> None:
        """
        Graceful shutdown of the task queue.
          1. Cancel periodic
          2. Cancel active tasks
          3. Unload
          4. Cleanup
        """
        self._check_stop_periodic_task()
        if self._periodic_task is not None:
            # wait for it to actually finish
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass

        if self._active_task is not None:
            if not self._active_task.done():
                logger.debug(f"Shutting down active task for {self.task_name}:{self.model_name}")
                self._active_task.cancel()
            try:
                await self._active_task
            except asyncio.CancelledError:
                pass
            self._active_task = None

        if self._load_task is not None:
            if not self._load_task.done():
                logger.debug(f"Shutting down load task for {self.task_name}:{self.model_name}")
                self._load_task.cancel()
            try:
                await self._load_task
            except asyncio.CancelledError:
                pass
            self._load_task = None

        self._unload_task()
        self._cleanup()

    async def fetch_result(
        self,
        payload_id: str,
        raise_when_error: bool=True
    ) -> Any:
        """
        Asynchronously wait for a job result. Non-blocking.
        If the job does not exist or is not started, this waits
        until it’s complete.

        If `raise_when_error=True`, will re-raise the exception
        if the underlying job had an error.
        """
        # Make sure the job is known or will be known
        while True:
            s = self._get_job_status(payload_id)
            if s in ("complete", "error"):
                break
            # queued or active
            await asyncio.sleep(0.01)

        # Return or raise the result
        result = self._job_results.get(payload_id, None)
        if isinstance(result, Exception) and raise_when_error:
            raise result

        return result

    async def fetch_callback_result(self, payload_id: str) -> Any:
        """
        Asynchronously wait for a job callback result. Non-blocking.
        If the job does not exist or is not started, this waits until it’s complete.
        """
        while True:
            s = self._get_job_status(payload_id)
            if s in ("complete", "error"):
                break
            # queued or active
            await asyncio.sleep(0.01)

        return self._job_callback_result.get(payload_id, None)

    def get_status(self, payload_id: str) -> TaskQueueResult:
        """
        Return a snapshot of the job’s status right now (non-blocking).
        """
        job_status = self._get_job_status(payload_id)
        job_result = self._job_results.get(payload_id, None)
        callback_result = self._job_callback_result.get(payload_id, None)

        return TaskQueueResult(
            id=payload_id,
            status=job_status,
            progress=self._get_job_progress(payload_id),
            rate=self._get_job_rate(payload_id),
            start=self._job_starts.get(payload_id),
            end=self._job_ends.get(payload_id),
            duration=self._get_job_duration(payload_id),
            remaining=self._get_job_remaining(payload_id),
            intermediate=self._get_job_intermediate(payload_id),
            result=job_result,
            callback=callback_result
        )

    """Dunder methods"""

    def __del__(self) -> None:
        """
        Deletes the task queue - best effort cleanup, prefer await shutdown.
        """
        self._check_stop_periodic_task()
        self._unload_task()
        self._cleanup()

    def __len__(self) -> int:
        """
        Returns the length of the queue.
        """
        return len(self._queue)

    def __contains__(self, payload_id: str) -> bool:
        """
        Checks if the payload ID is active, is in the queue, or has a result.
        """
        return (
            payload_id == self._active_id or
            any(p_id == payload_id for p_id, _ in self._queue) or
            payload_id in self._job_results
        )

    def __call__(self, **kwargs: Any) -> TaskQueueResult:
        """
        Calls the task queue.
        Either adds the job to the queue or returns the result of the job that matches the given arguments.
        When the job is running, this function will return the status of the job.
        """
        payload_id = kwargs.pop("id", None)
        callback = kwargs.pop("callback", None)

        if payload_id is None:
            payload_id = get_payload_id(kwargs)

        status = self._get_job_status(payload_id)
        if status == "new":
            if not kwargs:
                raise ValueError("No arguments provided for task!")

            if self.full:
                raise ValueError("Queue is full, cannot add job.")

            # Enqueue
            with self._lock:
                self._executions += 1
                self._queue.append((payload_id, kwargs))
                if callback is not None:
                    self._job_callback[payload_id] = callback

        # Return a snapshot of current status
        return self.get_status(payload_id)
