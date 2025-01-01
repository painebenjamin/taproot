from __future__ import annotations

import re
import asyncio

from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
    TYPE_CHECKING,
    cast,
)
from typing_extensions import Literal

from contextlib import contextmanager

from ..payload import *
from ..config import ExecutorConfig
from ..exceptions import CapacityExceededError
from ..util import (
    logger,
    parse_address,
    format_address,
    generate_id,
    get_metadata,
    get_continuation_depth,
    get_parameters_from_result,
)

from .config import ConfigServer

if TYPE_CHECKING:
    from ..tasks import TaskQueue

class Executor(ConfigServer):
    """
    The executor class that executes the task.
    """
    config_class = ExecutorConfig
    queue: TaskQueue

    """Configured attributes"""

    @property
    def max_continuation_depth(self) -> int:
        """
        Returns the maximum continuation depth.
        """
        configured = self.config.max_continuation_depth
        if configured is None:
            return 10
        return min(100, int(configured))

    @max_continuation_depth.setter
    def max_continuation_depth(self, value: int) -> None:
        """
        Sets the maximum continuation depth.
        """
        self.config.max_continuation_depth = value

    """Read-only attributes"""

    @property
    def task_name(self) -> str:
        """
        Returns the task name.
        """
        return str(self.config.queue_config.task)

    @property
    def model_name(self) -> Optional[str]:
        """
        Returns the model name.
        """
        configured = self.config.queue_config.model
        if configured is None:
            return None
        return str(configured)

    @property
    def queue_status(self) -> Dict[str, Any]:
        """
        Returns the status of the queue.
        """
        capacity = self.queue.capacity
        return {
            "status": self.queue.status,
            "activity": self.queue.activity,
            "capacity": capacity,
            "executions": self.queue.executions,
            "queued": len(self.queue),
        }

    @property
    def allocation(self) -> Literal["static", "dynamic"]:
        """
        Returns the allocation type.
        """
        return self.config.allocation # type: ignore[no-any-return]

    """Private methods"""

    def _send_continuation_payload(
        self,
        address: str,
        payload: TaskPayload
    ) -> None:
        """
        Sends a continuation payload to an executor.
        Kicks off the task but does not wait for the result.
        """
        logger.debug(f"Sending complete continuation payload to {address}: {payload}")
        client = self.get_client_for_address(address)
        asyncio.create_task(client(payload))
        return

    async def _send_continuation(
        self,
        overseer: str,
        continuation: Union[TaskPayload, List[TaskPayload]],
        result: Any,
        client_id: Optional[str]=None,
    ) -> Union[ExecutorTargetPayload, List[ExecutorTargetPayload]]:
        """
        Sends one or more continuations to an overseer.
        Continuations are reserved synchronously, then sent asynchronously.
        The client will receive the executor target and ID, and they will then
        need to send a request to that executor to get the status and results.
        """
        return_first = not isinstance(continuation, list)

        if not isinstance(continuation, list):
            continuation = [continuation]

        executor_futures: List[Coroutine[Any, Any, Any]] = []
        loop = asyncio.get_event_loop()
        parameters_list: List[Dict[str, Any]] = []
        for request in continuation:
            kwargs: Dict[str, Any] = {}
            parameters = request.get("parameters", None)
            if parameters is not None:
                kwargs.update(parameters)
            result_parameters = request.get("result_parameters", None)
            if result_parameters is not None:
                mapped_parameters = get_parameters_from_result(
                    result,
                    result_parameters
                )
                assert isinstance(mapped_parameters, dict), "Mapped parameters must be a dictionary."
                kwargs.update(mapped_parameters)

            parameters_list.append(kwargs)
            payload = {
                **request,
                **{
                    "parameters": get_metadata(kwargs),
                    "client_id": client_id,
                    "id": generate_id(),
                }
            }
            payload.pop("result_parameters", None) # Remove the result parameter map if present

            logger.debug(f"Reserving continuation for {overseer}: {payload}")

            overseer_client = self.get_client_for_address(overseer)
            overseer_client.address = overseer
            executor_futures.append(overseer_client(payload, timeout=0.5))

        executor_targets = loop.run_until_complete(
            asyncio.gather(*executor_futures),
        )

        for request, parameters, executor_target in zip(continuation, parameters_list, executor_targets):
            # Assemble the payload
            executor_payload = {
                **request,
                **{
                    "id": executor_target["id"],
                    "client_id": client_id,
                    "parameters": parameters,
                    "wait_for_result": False,
                    "overseer": overseer,
                }
            }
            executor_payload.pop("result_parameters", None) # Remove the result parameter map if present

            # If the address is just a path, make it relative to the overseer
            if not re.match(r"^\w+://", executor_target["address"]):
                address_parts = parse_address(overseer)
                # Replace last part of path with the executor target address
                target_address = executor_target["address"]
                if target_address.startswith("/"):
                    target_address = target_address[1:]
                address_path = address_parts.get("path", None)
                if address_path is None:
                    address_path = ""
                address_parts["path"] = address_path.rsplit("/", 1)[0] + "/" + target_address
                executor_target["address"] = format_address(address_parts)

            # Kick off the continuation
            self._send_continuation_payload(
                executor_target["address"],
                executor_payload # type: ignore[arg-type]
            )

        if return_first:
            return executor_targets[0] # type: ignore[no-any-return]
        return executor_targets

    """Overrides"""

    async def status(self, data: Any=None) -> ExecutorStatusPayload:
        """
        Returns the status of the executor.
        """
        status = cast(ExecutorStatusPayload, await super().status(data))
        status.update(self.queue_status) # type: ignore[typeddict-item]
        if data is not None and isinstance(data, str):
            status["has_id"] = data in self.queue
        status["allocation"] = self.allocation
        return status

    async def handle(self, request: Any) -> Any:
        """
        Handles the request by executing the task.
        """
        if not isinstance(request, dict):
            raise ValueError(f"Request must be a dictionary - got '{type(request).__name__}.'")

        wait_for_result = request.get("wait_for_result", True)

        request_id = request.get("id", None)
        client_id = request.get("client_id", None)
        return_metadata = request.get("return_metadata", False)

        if request_id is not None and client_id is not None:
            payload_id = f"{client_id}:{request_id}"
        else:
            payload_id = None

        capacity = self.queue.capacity

        if capacity <= 0:
            raise CapacityExceededError("Cannot execute request: Queue is full.")

        # look for continuations for callback
        continuation_message: Optional[str] = None
        callback: Optional[Callable[[Any], Any]] = None
        if request.get("continuation", None) is not None:
            overseer = request.get("overseer", None)
            if overseer is None:
                continuation_message = "Error: No overseer provided, cannot issue continuation."
            else:
                continuation_depth = get_continuation_depth(
                    request["continuation"],
                    self.max_continuation_depth
                )
                if continuation_depth > self.max_continuation_depth:
                    continuation_message = f"Error: Continuation depth exceeds maximum. Request has depth of {continuation_depth}, maximum is {self.max_continuation_depth}."
                else:
                    callback = lambda result: self._send_continuation( 
                        overseer,
                        continuation=request["continuation"],
                        result=result,
                        client_id=client_id,
                    )

        kwargs: Dict[str, Any] = {
            "id": payload_id,
            "callback": callback,
            "wait_for_result": wait_for_result
        }
        request_parameters = request.get("parameters", {})
        if isinstance(request_parameters, dict):
            kwargs.update(request_parameters)

        result = self.queue(**kwargs)
        result["id"] = request_id # Remove the client ID from the result
        result_callback = result.pop("callback", None) # type: ignore[misc]
        if result_callback is not None:
            result["continuation"] = result_callback # type: ignore[typeddict-unknown-key]
        elif continuation_message is not None:
            result["continuation"] = continuation_message # type: ignore[typeddict-unknown-key]

        if wait_for_result and not return_metadata:
            return result["result"] # Return only the result
        return result

    @contextmanager
    def context(self) -> Iterator[None]:
        """
        Returns a context manager for the executor.
        """
        from ..tasks import Task, TaskQueue
        with super().context():
            if self.config.install:
                task = Task.get(
                    task=self.config.queue_config.task,
                    model=self.config.queue_config.model,
                    available_only=False,
                    model_dir=self.config.queue_config.task_config.model_dir,
                
                )
                if task is None:
                    if self.config.queue_config.model is not None:
                        raise RuntimeError(f"Model '{self.config.queue_config.task}:{self.config.queue_config.model}' not found.")
                    raise RuntimeError(f"Task '{self.config.queue_config.task}' not found.")
                task.ensure_availability(
                    text_callback=logger.debug,
                )
            self.queue = TaskQueue(self.config.queue_config)
            yield

    def shutdown(self) -> None:
        """
        Shuts down the executor.
        """
        self.queue.shutdown()
        super().shutdown()

    """Additional Public methods"""

    async def has_id(
        self,
        client_id: str,
        request_id: str,
        timeout: Optional[float]=0.1,
        retries: int=0
    ) -> bool:
        """
        Returns whether the executor has the ID.
        """
        return bool(
            (
                await self.get_status(
                    data=f"{client_id}:{request_id}",
                    timeout=timeout,
                    retries=retries
                )
            ).get("has_id", False)
        )
