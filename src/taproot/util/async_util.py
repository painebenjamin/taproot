from __future__ import annotations

import platform
import asyncio
import signal

from typing import Any, Callable, Awaitable, List, Optional, Coroutine, TYPE_CHECKING

from .log_util import logger

if TYPE_CHECKING:
    from ..server.base import Server

CallbackType = Callable[..., Optional[Awaitable[Any]]]

__all__ = [
    "aioconsole_is_available",
    "uvloop_is_available",
    "execute_and_await",
    "AsyncRunner",
    "TaskRunner",
    "ServerRunner",
]

UVLOOP_AVAILABLE: Optional[bool] = None
def uvloop_is_available() -> bool:
    """
    Returns True if the uvloop is available.

    This is only available on Unix systems, and is recommended for
    performance reasons. If it's not available, we'll print a warning
    if it's recommended to install it.
    """
    global UVLOOP_AVAILABLE
    if UVLOOP_AVAILABLE is None:
        try:
            import uvloop
            UVLOOP_AVAILABLE = True
        except ImportError:
            if platform.system() != "Windows":
                logger.warning("uvloop is not installed. It's recommended to install it with `pip install uvloop`.")
            UVLOOP_AVAILABLE = False
    return UVLOOP_AVAILABLE

def aioconsole_is_available() -> bool:
    """
    Returns True if aioconsole is available.

    When this is needed it's essential, so we'll print an error instead
    of a warning like we do with uvloop.
    """
    try:
        import aioconsole # type: ignore[import-untyped,import-not-found,unused-ignore]
        return bool(aioconsole)
    except ImportError:
        return False

async def execute_and_await(
    method: Callable[..., Any],
    *args: Any,
    **kwargs: Any
) -> Any:
    """
    Executes a method and awaits the result if it's a coroutine.

    :param method: The method to execute
    :param args: The arguments to pass to the method
    :param kwargs: The keyword arguments to pass to the method
    :return: The result of the method
    """
    result = method(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result

class AsyncRunner:
    """
    A class for running multiple async callables in parallel or sequentially.
    """
    def __init__(
        self,
        *callables: Callable[..., Awaitable[Any]],
        sequential: bool = False,
    ) -> None:
        self.callables = list(callables)
        self.sequential = sequential

    async def main(self) -> None:
        """
        Runs all callables, either in parallel or sequentially.
        """
        if self.sequential:
            for method in self.callables:
                await method()
        else:
            await asyncio.gather(*[
                method()
                for method in self.callables
            ])

    def run(self, debug: bool=False, ignore_cancel: bool=True) -> None:
        """
        Runs the main method in the event loop.
        """
        try:
            if uvloop_is_available():
                import uvloop
                uvloop.run(self.main(), debug=debug)
            else:
                asyncio.run(self.main())
        except asyncio.CancelledError:
            if not ignore_cancel:
                raise

class TaskRunner:
    """
    A context manager for starting and stopping tasks.
    """
    task: Optional[asyncio.Task[Any]]

    def __init__(self, coro: Coroutine[Any, Any, Any]) -> None:
        """
        :param coro: The coroutine to run
        """
        self.coro = coro
        self.task = None

    async def __aenter__(self) -> asyncio.Task[Any]:
        """
        On entering the context manager, start the task.
        """
        loop = asyncio.get_event_loop()
        self.task = loop.create_task(self.coro)
        await asyncio.sleep(0.01) # Sleep briefly
        return self.task

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        On exiting the context manager, stop the task.
        """
        if self.task is not None:
            if not self.task.done():
                self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

class ServerRunner:
    """
    Runs multiple servers using UVLoop, handling
    graceful shutdowns and restarts.
    """
    before_start_callbacks: List[CallbackType]
    after_start_callbacks: List[CallbackType]
    before_stop_callbacks: List[CallbackType]
    after_stop_callbacks: List[CallbackType]

    def __init__(self, *servers: Server) -> None:
        self.servers = list(servers)
        self.before_start_callbacks = []
        self.after_start_callbacks = []
        self.before_stop_callbacks = []
        self.after_stop_callbacks = []

    def before_start(self, callback: CallbackType) -> None:
        """
        Registers a callback that will be executed before the server starts.
        """
        self.before_start_callbacks.append(callback)

    def after_start(self, callback: CallbackType) -> None:
        """
        Registers a callback that will be executed after the server starts.
        """
        self.after_start_callbacks.append(callback)

    def before_stop(self, callback: CallbackType) -> None:
        """
        Registers a callback that will be executed before the server stops.
        """
        self.before_stop_callbacks.append(callback)

    def after_stop(self, callback: CallbackType) -> None:
        """
        Registers a callback that will be executed after the server stops.
        """
        self.after_stop_callbacks.append(callback)

    async def main(self, install_signal_handlers: bool=True) -> None:
        """
        The main loop that runs the servers.
        """
        for callback in self.before_start_callbacks:
            await execute_and_await(callback)

        exit_event = asyncio.Event()

        if install_signal_handlers:
            def exit_handler(signum: int, frame: Any) -> None:
                exit_event.set()

            signal.signal(signal.SIGINT, exit_handler)
            signal.signal(signal.SIGTERM, exit_handler)

        loop = asyncio.get_event_loop()

        # Start all the servers
        tasks: List[asyncio.Task[Any]] = []
        for server in self.servers:
            task = loop.create_task(server.run())
            tasks.append(task)

        # Sleep briefly
        await asyncio.sleep(0.01)

        try:
            # Assert connectivity on all of them in parallel
            await asyncio.gather(*[
                server.assert_connectivity()
                for server in self.servers
            ])

            # Call the after start callbacks
            for callback in self.after_start_callbacks:
                await execute_and_await(callback)

            # Wait for the exit event
            while not exit_event.is_set():
                # If all tasks are done, break
                if all(task.done() for task in tasks):
                    break
                try:
                    await asyncio.wait_for(exit_event.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass
        finally:
            # Call the before stop callbacks
            for callback in self.before_stop_callbacks:
                await execute_and_await(callback)

            # Shutdown all the remaining servers in parallel
            await asyncio.gather(*[
                server.exit()
                for task, server in zip(tasks, self.servers)
                if not task.done()
            ])

            # Wait one second
            await asyncio.sleep(1.0)

            # Ensure all tasks are done
            for task in tasks:
                if not task.done():
                    task.cancel()

            # Await all tasks
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                pass

            # Call the after stop callbacks
            for callback in self.after_stop_callbacks:
                await execute_and_await(callback)

    def run(
        self,
        install_signal_handlers: bool=True,
        debug: bool=False
    ) -> None:
        """
        Executes the main loop.
        """
        if uvloop_is_available():
            import uvloop
            uvloop.run(
                self.main(install_signal_handlers=install_signal_handlers),
                debug=debug
            )
        else:
            asyncio.run(
                self.main(install_signal_handlers=install_signal_handlers)
            )
