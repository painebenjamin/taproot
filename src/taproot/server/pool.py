from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future
from typing import Any, Callable

__all__ = ["TrackingProcessPoolExecutor", "TrackingThreadPoolExecutor"]

class TrackingProcessPoolExecutor(ProcessPoolExecutor):
    """
    A ProcessPoolExecutor that tracks the number of submitted tasks and
    provides a method to check if there are available workers.
    """
    _max_workers: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._submitted_tasks = 0

    def submit(self, fn: Callable[[Any], Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        """
        Submit a task to the pool. The task is wrapped in a Future object
        and the number of submitted tasks is incremented. A callback is
        added to the Future object to decrement the number of submitted
        tasks when the task is done.
        """
        self._submitted_tasks += 1
        future = super().submit(fn, *args, **kwargs)
        future.add_done_callback(self._task_done_callback)
        return future

    def _task_done_callback(self, future: Future[Any]) -> None:
        """
        Callback to decrement the number of submitted tasks when the task
        is done.
        """
        self._submitted_tasks -= 1

    @property
    def has_available_workers(self) -> bool:
        """
        Check if there are available workers in the pool.
        """
        return self._submitted_tasks < self._max_workers

class TrackingThreadPoolExecutor(ThreadPoolExecutor):
    """
    A ThreadPoolExecutor that tracks the number of submitted tasks and
    provides a method to check if there are available workers.
    """
    _max_workers: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._submitted_tasks = 0

    def submit(self, fn: Callable[[Any], Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        """
        Submit a task to the pool. The task is wrapped in a Future object
        and the number of submitted tasks is incremented. A callback is
        added to the Future object to decrement the number of submitted
        tasks when the task is done.
        """
        self._submitted_tasks += 1
        future = super().submit(fn, *args, **kwargs)
        future.add_done_callback(self._task_done_callback)
        return future

    def _task_done_callback(self, future: Future[Any]) -> None:
        """
        Callback to decrement the number of submitted tasks when the task
        is done.
        """
        self._submitted_tasks -= 1

    @property
    def has_available_workers(self) -> bool:
        """
        Check if there are available workers in the pool.
        """
        return self._submitted_tasks < self._max_workers
