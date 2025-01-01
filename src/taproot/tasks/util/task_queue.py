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
from taproot.util import logger, get_payload_id

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
        self._start_tasks()

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

    """Internal methods"""

    def _start_tasks(self) -> None:
        """
        Starts the initial rounds of tasks.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        logger.info(f"Starting load task for {self.task_name}:{self.model_name}")
        self._load_task = loop.create_task(self._initialize_task())

        logger.info(f"Starting periodic task for {self.task_name}:{self.model_name}")
        self._periodic_task = loop.create_task(self._periodic_check())
        weakref.finalize(self, self._check_stop_periodic_task, self._periodic_task)

    def _check_start_periodic_task(self) -> None:
        """
        Checks if the periodic task should be started.
        """
        if self.zombie:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            logger.info(f"Starting periodic task for {self.task_name}:{self.model_name}")
            self._periodic_task = loop.create_task(self._periodic_check())
            weakref.finalize(self, self._check_stop_periodic_task, self._periodic_task)
        return

    def _check_stop_periodic_task(self, *args: Any) -> None:
        """
        Checks if the periodic task should be stopped.
        """
        if self._periodic_task is not None:
            if not self._periodic_task.done():
                logger.info(f"Stopping periodic task for {self.task_name}:{self.model_name}")
                self._periodic_task.cancel()
                asyncio.get_event_loop().create_task(self._finalize_periodic_task(self._periodic_task))
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

    def _start_next_job(self) -> None:
        """
        Starts the next job in the queue.
        This assumes that the existing job has already been checked for completion.
        """
        if self._queue:
            payload_id, payload = self._queue.popleft()
            self._active_id = payload_id
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

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
                task_cls = Task.get(
                    task=self.task_name,
                    model=self.model_name
                )
                if task_cls is None:
                    raise ValueError(f"Task {self.task_name}:{self.model_name} not found.")
                task_instance = task_cls(self.task_config) # type: ignore[arg-type]
                task_instance.load()
                self._task = task_instance
            except Exception as ex:
                logger.error(f"Error initializing task {self.task_name}:{self.model_name}: {type(ex).__name__} {str(ex)}")
                raise
        return

    async def _wait_for_task(self, polling_interval: float=0.01) -> None:
        """
        Waits for the task to be initialized.
        """
        while not hasattr(self, "_task"):
            await asyncio.sleep(polling_interval)
        return

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
        await self._wait_for_task()
        self._task.num_steps = 1 # Reset steps and counters
        self._job_starts[payload_id] = perf_counter()
        result = await asyncio.to_thread(self._execute_task, kwargs)
        callback = self._job_callback.pop(payload_id, None)
        callback_result: Any = None
        if callback is not None:
            callback_result = await callback(result)

        with self._lock:
            self._job_results[payload_id] = result
            self._job_ends[payload_id] = perf_counter()
            self._job_access[payload_id] = self._job_ends[payload_id]
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
            self._job_results.pop(payload_id)
            self._job_starts.pop(payload_id)
            self._job_ends.pop(payload_id)
            self._job_access.pop(payload_id)
            self._job_callback.pop(payload_id, None)
            self._job_callback_result.pop(payload_id, None)

    def _has_result(self, payload_id: str) -> bool:
        """
        Checks if the result is in the results dictionary.
        """
        return payload_id in self._job_results

    def _has_queued_job(self, payload_id: str) -> bool:
        """
        Checks if the job is in the queue.
        """
        return payload_id in [
            queued_payload_id for
            queued_payload_id, queued_args
            in self._queue
        ]

    def _is_active_job(self, payload_id: str) -> bool:
        """
        Checks if the job is the active job.
        """
        return payload_id == self._active_id

    def _get_job_status(self, payload_id: str) -> Literal["new", "queued", "active", "complete", "error"]:
        """
        Returns the status of the job.
        """
        if self._has_result(payload_id):
            if isinstance(self._job_results[payload_id], Exception):
                return "error"
            return "complete"
        elif self._is_active_job(payload_id):
            return "active"
        elif self._has_queued_job(payload_id):
            return "queued"
        return "new"

    def _get_job_progress(self, payload_id: str) -> float:
        """
        Returns the progress of the job.
        """
        if self._has_result(payload_id):
            return 1.0
        elif self._is_active_job(payload_id):
            return self._task.progress
        return 0.0

    def _get_job_rate(self, payload_id: str) -> Optional[float]:
        """
        Returns the rate of the job.
        """
        if self._has_result(payload_id):
            job_start = self._job_starts.get(payload_id, None)
            job_end = self._job_ends.get(payload_id, None)
            if job_start is not None and job_end is not None:
                return 1.0 / (job_end - job_start)
        elif self._is_active_job(payload_id):
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
        elif job_start is not None:
            return perf_counter() - job_start
        return None

    def _get_job_remaining(self, payload_id: str) -> Optional[float]:
        """
        Returns the remaining time of the job.
        """
        if self._has_result(payload_id):
            return 0.0
        elif self._is_active_job(payload_id):
            return self._task.remaining
        return None

    def _get_job_intermediate(self, payload_id: str) -> Optional[Any]:
        """
        Returns the remaining time of the job.
        """
        if self._is_active_job(payload_id):
            return self._task.last_intermediate
        return None

    def _get_callback_result(self, payload_id: str) -> Optional[Any]:
        """
        Returns the callback result of the job.
        """
        return self._job_callback_result.get(payload_id, None)

    def _get_job_result(
        self,
        payload_id: str,
        wait: bool=False,
        raise_when_unfinished: bool=False,
        polling_interval: float=0.01
    ) -> Any:
        """
        Returns the result of the job.
        """
        with self._lock:
            if self._has_result(payload_id):
                # Reset the time when the result is accessed
                logger.debug(f"Result for {payload_id} found.")
                self._job_access[payload_id] = perf_counter()
                return self._job_results[payload_id]
            if not wait:
                logger.debug(f"Result for {payload_id} not found.")
                if raise_when_unfinished:
                    raise ValueError(f"Result for {payload_id} not found.")
                return None

        logger.debug(f"Waiting for result for {payload_id}.")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        while True:
            loop.run_until_complete(asyncio.sleep(polling_interval))
            with self._lock:
                if self._has_result(payload_id):
                    logger.debug(f"Result for {payload_id} found.")
                    return self._job_results[payload_id]

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
        if hasattr(self, "_task"):
            self._task.unload()

    def _offload_task(self) -> None:
        """
        Offloads the task (from GPU to CPU).
        """
        if hasattr(self, "_task"):
            self._task.offload()

    def _onload_task(self) -> None:
        """
        Onloads the task (from CPU to GPU).
        """
        if hasattr(self, "_task"):
            self._task.onload()

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

    def shutdown(self) -> None:
        """
        Shuts down the task queue.
        """
        self._check_stop_periodic_task()
        self._unload_task()

    """Public methods"""

    def __del__(self) -> None:
        """
        Deletes the task queue.
        """
        self._check_stop_periodic_task()
        self._unload_task()

    def __len__(self) -> int:
        """
        Returns the length of the queue.
        """
        return len(self._queue)

    def __contains__(self, payload_id: str) -> bool:
        """
        Checks if the payload ID is active, is in the queue, or has a result.
        """
        return self._has_result(payload_id) or self._has_queued_job(payload_id) or self._is_active_job(payload_id)

    def __call__(self, **kwargs: Any) -> TaskQueueResult:
        """
        Calls the task queue.
        Either adds the job to the queue or returns the result of the job that matches the given arguments.
        When the job is running, this function will return the status of the job.
        """
        wait_for_result: bool = kwargs.pop("wait_for_result", False)
        raise_error_result: bool = kwargs.pop("raise_error_result", False)
        payload_id = kwargs.pop("id", None)
        callback = kwargs.pop("callback", None)

        if payload_id is None:
            payload_id = get_payload_id(kwargs)

        # Get the task result first, if the call waits for the result
        # then the job status will change after this is called
        initial_status = self._get_job_status(payload_id)
        if initial_status == "new":
            if not kwargs:
                raise ValueError("No arguments provided for task!")

            logger.debug(f"Adding job for {self.task_name}:{self.model_name} with ID {payload_id}")
            self._add_job(
                payload_id,
                callback=callback,
                **kwargs
            )

        # Now get the task result
        task_result = self._get_job_result(
            payload_id,
            wait=wait_for_result,
            raise_when_unfinished=False
        )
        callback_result = self._get_callback_result(payload_id)

        task_queue_result: TaskQueueResult = {
            "id": payload_id,
            "status": initial_status if not wait_for_result else self._get_job_status(payload_id),
            "progress": self._get_job_progress(payload_id),
            "rate": self._get_job_rate(payload_id),
            "start": self._job_starts.get(payload_id, None),
            "end": self._job_ends.get(payload_id, None),
            "duration": self._get_job_duration(payload_id),
            "remaining": self._get_job_remaining(payload_id),
            "intermediate": self._get_job_intermediate(payload_id),
            "result": task_result,
            "callback": callback_result
        }
        if task_queue_result["status"] == "error" and raise_error_result:
            raise task_result
        return task_queue_result
