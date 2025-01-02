from __future__ import annotations

import asyncio
import traceback

from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterator,
    List,
    Literal,
    Literal,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING
)

from contextlib import contextmanager, ExitStack

from ..constants import *
from ..payload import *
from ..util import (
    find_free_memory_port,
    find_free_unix_socket,
    find_free_port,
    pack_control_message,
    get_parameter_metadata,
    get_payload_id,
    parse_address,
    is_absolute_address,
    get_absolute_address_from_relative,
    generate_id,
    logger,
)
from ..config import *
from .base import Client

if TYPE_CHECKING:
    from ..server import Dispatcher

class Tap(ConfigMixin):
    """
    Taps into a taproot cluster.
    """
    config_class = TapConfig
    exit_stack: ExitStack

    """Private properties"""
    _executor_addresses: Dict[str, str]

    """Default properties"""

    @property
    def default_use_local(self) -> bool:
        """
        Returns whether to use the local dispatcher by default.
        """
        return self.config.local is not None

    @property
    def default_local_use_encryption(self) -> bool:
        """
        Returns whether to use encryption for the local dispatcher by default.
        """
        return self.config.local.encryption is not None

    @property
    def default_use_remote(self) -> bool:
        """
        Returns whether to use the local dispatcher by default.
        """
        return self.config.remote is not None

    @property
    def default_remote_use_encryption(self) -> bool:
        """
        Returns whether to use encryption for the remote dispatcher by default.
        """
        return self.config.remote.encryption is not None

    """Local getters/setters"""

    @property
    def use_local(self) -> bool:
        """
        Returns whether to use the local dispatcher.
        """
        if not hasattr(self, "_use_local"):
            self._use_local = self.default_use_local
        return self._use_local

    @use_local.setter
    def use_local(self, value: bool) -> None:
        """
        Sets whether to use the local dispatcher.
        """
        self._use_local = value

    @property
    def local_host(self) -> Optional[str]:
        """
        Returns the local dispatcher host.
        """
        return self.config.local.host if self.config.local else None

    @local_host.setter
    def local_host(self, value: Optional[str]) -> None:
        """
        Sets the local dispatcher host.
        """
        if not self.config.local:
            self.config.local = DispatcherConfig()
        self.config.local.host = value

    @property
    def local_port(self) -> Optional[int]:
        """
        Returns the local dispatcher port.
        """
        return self.config.local.port if self.config.local else None

    @local_port.setter
    def local_port(self, value: Optional[int]) -> None:
        """
        Sets the local dispatcher port.
        """
        if not self.config.local:
            self.config.local = DispatcherConfig()
        self.config.local.port = value

    @property
    def local_protocol(self) -> Optional[PROTOCOL_LITERAL]:
        """
        Returns the local dispatcher protocol.
        """
        return self.config.local.protocol if self.config.local else None

    @local_protocol.setter
    def local_protocol(self, value: PROTOCOL_LITERAL) -> None:
        """
        Sets the local dispatcher protocol.
        """
        if not self.config.local:
            self.config.local = DispatcherConfig()
        self.config.local.protocol = value

    @property
    def local_scheme(self) -> Optional[SCHEME_LITERAL]:
        """
        Returns the local dispatcher scheme.
        """
        if not self.config.local:
            return None
        if self.local_protocol == "tcp" and self.local_use_encryption:
            return "tcps"
        elif self.local_protocol == "ws" and self.local_use_encryption:
            return "wss"
        return self.local_protocol

    @local_scheme.setter
    def local_scheme(self, value: Optional[SCHEME_LITERAL]) -> None:
        """
        Sets the local dispatcher scheme.
        """
        if not self.config.local:
            self.config.local = DispatcherConfig()
        if value == "tcps":
            self.local_protocol = "tcp"
            self.local_use_encryption = True
        elif value == "wss":
            self.local_protocol = "ws"
            self.local_use_encryption = True
        else:
            self.local_protocol = value
            self.local_use_encryption = False

    @property
    def local_encryption_key_length(self) -> Optional[int]:
        """
        Returns the local dispatcher encryption key length.
        """
        return self.config.local.encryption.encryption_key_length if self.config.local and self.config.local.encryption else None

    @local_encryption_key_length.setter
    def local_encryption_key_length(self, value: int) -> None:
        """
        Sets the local dispatcher encryption key length.
        """
        if not self.config.local:
            self.config.local = DispatcherConfig()
        if not self.config.local.encryption:
            self.config.local.encryption = EncryptionConfig()
        self.config.local.encryption.encryption_key_length = value

    @property
    def local_encryption_key(self) -> Optional[str]:
        """
        Returns the local dispatcher encryption key.
        """
        return self.config.local.encryption.encryption_key if self.config.local and self.config.local.encryption else None

    @local_encryption_key.setter
    def local_encryption_key(self, value: str) -> None:
        """
        Sets the local dispatcher encryption key.
        """
        if not self.config.local:
            self.config.local = DispatcherConfig()
        if not self.config.local.encryption:
            self.config.local.encryption = EncryptionConfig()
        self.config.local.encryption.encryption_key = value

    @property
    def local_encryption_use_aesni(self) -> Optional[bool]:
        """
        Returns whether to use AESNI for the local dispatcher encryption.
        """
        return self.config.local.encryption.encryption_use_aesni if self.config.local and self.config.local.encryption else None

    @local_encryption_use_aesni.setter
    def local_encryption_use_aesni(self, value: bool) -> None:
        """
        Sets whether to use AESNI for the local dispatcher encryption.
        """
        if not self.config.local:
            self.config.local = DispatcherConfig()
        if not self.config.local.encryption:
            self.config.local.encryption = EncryptionConfig()
        self.config.local.encryption.encryption_use_aesni = value

    @property
    def local_use_encryption(self) -> bool:
        """
        Returns whether to use encryption for the local dispatcher.
        """
        if not hasattr(self, "_local_use_encryption"):
            self._local_use_encryption = self.default_local_use_encryption
        return self._local_use_encryption

    @local_use_encryption.setter
    def local_use_encryption(self, value: bool) -> None:
        """
        Sets whether to use encryption for the local dispatcher.
        """
        self._local_use_encryption = value

    @property
    def local_certfile(self) -> Optional[str]:
        """
        Returns the local dispatcher certfile.
        """
        return self.config.local.certfile if self.config.local else None

    @local_certfile.setter
    def local_certfile(self, value: Optional[str]) -> None:
        """
        Sets the local dispatcher certfile.
        """
        if not self.config.local:
            self.config.local = DispatcherConfig()
        self.config.local.certfile = value

    @property
    def local_keyfile(self) -> Optional[str]:
        """
        Returns the local dispatcher keyfile.
        """
        return self.config.local.keyfile if self.config.local else None

    @local_keyfile.setter
    def local_keyfile(self, value: Optional[str]) -> None:
        """
        Sets the local dispatcher keyfile.
        """
        if not self.config.local:
            self.config.local = DispatcherConfig()
        self.config.local.keyfile = value

    @property
    def local_address(self) -> Optional[str]:
        """
        Returns the local dispatcher address.
        """
        if not self.use_local or not self.local_scheme:
            return None
        base = f"{self.local_scheme}://"
        if self.local_scheme == "memory":
            return f"{base}{self.local_port}"
        elif self.local_scheme == "unix":
            return f"{base}{self.local_host}"
        return f"{base}{self.local_host}:{self.local_port}"

    @local_address.setter
    def local_address(self, value: str) -> None:
        """
        Sets the local dispatcher address.
        """
        address = parse_address(value)
        self.local_scheme = address["scheme"]
        if address["host"]:
            self.local_host = address["host"]
        if address["port"]:
            self.local_port = address["port"]

    """Remote getters/setters"""

    @property
    def use_remote(self) -> bool:
        """
        Returns whether to use the remote dispatcher.
        """
        if not hasattr(self, "_use_remote"):
            self._use_remote = self.default_use_remote
        return self._use_remote

    @use_remote.setter
    def use_remote(self, value: bool) -> None:
        """
        Sets whether to use the remote dispatcher.
        """
        self._use_remote = value

    @property
    def remote_host(self) -> Optional[str]:
        """
        Returns the remote dispatcher host.
        """
        return self.config.remote.host if self.config.remote else None

    @remote_host.setter
    def remote_host(self, value: Optional[str]) -> None:
        """
        Sets the remote dispatcher host.
        """
        if not self.config.remote:
            self.config.remote = ClientConfig()
        self.config.remote.host = value

    @property
    def remote_port(self) -> Optional[int]:
        """
        Returns the remote dispatcher port.
        """
        return self.config.remote.port if self.config.remote else None

    @remote_port.setter
    def remote_port(self, value: Optional[int]) -> None:
        """
        Sets the remote dispatcher port.
        """
        if not self.config.remote:
            self.config.remote = ClientConfig()
        self.config.remote.port = value

    @property
    def remote_path(self) -> Optional[str]:
        """
        Returns the remote dispatcher path.
        """
        return self.config.remote.path if self.config.remote else None

    @remote_path.setter
    def remote_path(self, value: Optional[str]) -> None:
        """
        Sets the remote dispatcher path.
        """
        if not self.config.remote:
            self.config.remote = ClientConfig()
        self.config.remote.path = value

    @property
    def remote_protocol(self) -> Optional[Literal["memory", "tcp", "unix", "ws"]]:
        """
        Returns the remote dispatcher protocol.
        """
        return self.config.remote.protocol if self.config.remote else None

    @remote_protocol.setter
    def remote_protocol(self, value: Literal["memory", "tcp", "unix", "ws"]) -> None:
        """
        Sets the remote dispatcher protocol.
        """
        if not self.config.remote:
            self.config.remote = ClientConfig()
        self.config.remote.protocol = value

    @property
    def remote_scheme(self) -> Optional[Literal["memory", "tcp", "tcps", "unix", "ws", "wss"]]:
        """
        Returns the remote dispatcher scheme.
        """
        if not self.config.remote:
            return None
        if self.remote_protocol == "tcp" and self.remote_use_encryption:
            return "tcps"
        if self.remote_protocol == "ws" and self.remote_use_encryption:
            return "wss"
        return self.remote_protocol

    @remote_scheme.setter
    def remote_scheme(self, value: Literal["memory", "tcp", "tcps", "unix", "ws", "wss"]) -> None:
        """
        Sets the remote dispatcher scheme.
        """
        if not self.config.remote:
            self.config.remote = ClientConfig()
        if value == "tcps":
            self.remote_protocol = "tcp"
            self.remote_use_encryption = True
        elif value == "wss":
            self.remote_protocol = "ws"
            self.remote_use_encryption = True
        else:
            self.remote_protocol = value
            self.remote_use_encryption = False

    @property
    def remote_encryption_key_length(self) -> Optional[int]:
        """
        Returns the remote dispatcher encryption key length.
        """
        return self.config.remote.encryption.encryption_key_length if self.config.remote and self.config.remote.encryption else None

    @remote_encryption_key_length.setter
    def remote_encryption_key_length(self, value: int) -> None:
        """
        Sets the remote dispatcher encryption key length.
        """
        if not self.config.remote:
            self.config.remote = ClientConfig()
        if not self.config.remote.encryption:
            self.config.remote.encryption = EncryptionConfig()
        self.config.remote.encryption.encryption_key_length = value

    @property
    def remote_encryption_key(self) -> Optional[bytes]:
        """
        Returns the remote dispatcher encryption key.
        """
        return (
            self.config.remote.encryption.encryption_key.encode("utf-8")
            if self.config.remote
            and self.config.remote.encryption else None
        )

    @remote_encryption_key.setter
    def remote_encryption_key(self, value: Union[str, bytes]) -> None:
        """
        Sets the remote dispatcher encryption key.
        """
        if not self.config.remote:
            self.config.remote = ClientConfig()
        if not self.config.remote.encryption:
            self.config.remote.encryption = EncryptionConfig()
        self.config.remote.encryption.encryption_key = value.decode("utf-8") if isinstance(value, bytes) else value

    @property
    def remote_encryption_use_aesni(self) -> Optional[bool]:
        """
        Returns whether to use AESNI for the remote dispatcher encryption.
        """
        return self.config.remote.encryption.encryption_use_aesni if self.config.remote and self.config.remote.encryption else None

    @remote_encryption_use_aesni.setter
    def remote_encryption_use_aesni(self, value: bool) -> None:
        """
        Sets whether to use AESNI for the remote dispatcher encryption.
        """
        if not self.config.remote:
            self.config.remote = ClientConfig()
        if not self.config.remote.encryption:
            self.config.remote.encryption = EncryptionConfig()
        self.config.remote.encryption.encryption_use_aesni = value

    @property
    def remote_use_encryption(self) -> bool:
        """
        Returns whether to use encryption for the remote dispatcher.
        """
        if not hasattr(self, "_remote_use_encryption"):
            self._remote_use_encryption = self.default_remote_use_encryption
        return self._remote_use_encryption

    @remote_use_encryption.setter
    def remote_use_encryption(self, value: bool) -> None:
        """
        Sets whether to use encryption for the remote dispatcher.
        """
        self._remote_use_encryption = value

    @property
    def remote_certfile(self) -> Optional[str]:
        """
        Returns the remote dispatcher certfile.
        """
        if not self.config.remote or not self.config.remote.encryption:
            return None
        return self.config.remote.encryption.certfile # type: ignore[no-any-return]

    @remote_certfile.setter
    def remote_certfile(self, value: Optional[str]) -> None:
        """
        Sets the remote dispatcher certfile.
        """
        if not self.config.remote:
            self.config.remote = ClientConfig()
        if not self.config.remote.encryption:
            self.config.remote.encryption = EncryptionConfig()
        self.config.remote.encryption.certfile = value

    @property
    def remote_address(self) -> Optional[str]:
        """
        Returns the remote dispatcher address.
        """
        if not self.use_remote:
            return None
        base = f"{self.remote_scheme}://"
        if self.remote_scheme == "memory":
            return f"{base}{self.remote_port}"
        elif self.remote_scheme == "unix":
            return f"{base}{self.remote_path}"
        path = "" if not self.remote_path else self.remote_path
        host_port = f"{self.remote_host}:{self.remote_port}" if self.remote_port else self.remote_host
        return f"{base}{host_port}{path}"

    @remote_address.setter
    def remote_address(self, value: str) -> None:
        """
        Sets the remote dispatcher address.
        """
        address = parse_address(value)
        self.remote_scheme = address["scheme"]
        self.remote_path = address["path"]
        if address["host"]:
            self.remote_host = address["host"]
        if address["port"]:
            self.remote_port = address["port"]
        elif self.remote_scheme == "wss":
            self.remote_port = 443
        elif self.remote_scheme == "ws":
            self.remote_port = 80

    """Getters Only"""

    @property
    def client_id(self) -> str:
        """
        Returns the client ID.
        """
        if not hasattr(self, "_client_id"):
            self._client_id = generate_id()
        return self._client_id

    @property
    def executor_addresses(self) -> Dict[str, str]:
        """
        Returns a dictionary of executor addresses.
        These will be re-used when calling tasks.
        """
        if not hasattr(self, "_executor_addresses"):
            self._executor_addresses = {}
        return self._executor_addresses

    """Context"""

    def __enter__(self) -> Tap:
        """
        Enters the context.
        """
        self.exit_stack = ExitStack()
        if self.use_local:
            dispatcher = self._get_local_dispatcher()
            self.exit_stack.enter_context(dispatcher) # type: ignore[arg-type]
            dispatcher.wait_for_connectivity()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        Exits the context.
        """
        if getattr(self, "exit_stack", None):
            self.exit_stack.close()

    """Private methods"""

    def _get_local_dispatcher(self) -> Dispatcher:
        """
        Gets the local dispatcher.
        """
        if not self.use_local or not self.config.local:
            raise ValueError("Local dispatcher is not enabled.")
        from ..server import Dispatcher
        return Dispatcher(self.config.local)

    def _get_client_for_address(self, address: str, use_local_encryption_config: bool=False) -> Client:
        """
        Gets an executor client.
        """
        client = Client()
        if not is_absolute_address(address) and self.remote_address is not None:
            client.address = get_absolute_address_from_relative(
                absolute_address=self.remote_address,
                relative_address=address,
            )
        else:
            client.address = address

        if client.use_encryption:
            if use_local_encryption_config and self.config.local and self.config.local.encryption:
                client.certfile = self.config.local.encryption.certfile
                client.encryption_key = self.config.local.encryption.encryption_key
                client.encryption_use_aesni = self.config.local.encryption.encryption_use_aesni
            elif not use_local_encryption_config and self.config.remote and self.config.remote.encryption:
                client.certfile = self.config.remote.encryption.certfile
                client.encryption_key = self.config.remote.encryption.encryption_key
                client.encryption_use_aesni = self.config.remote.encryption.encryption_use_aesni

        return client

    def _get_remote_client(self) -> Client:
        """
        Returns the remote dispatcher client.
        """
        if not self.use_remote or not self.config.remote:
            raise ValueError("Remote dispatcher is not enabled.")
        client = Client()
        client.scheme = self.remote_scheme # type: ignore[assignment]
        client.path = self.remote_path
        if self.remote_host:
            client.host = self.remote_host
        if self.remote_port:
            client.port = self.remote_port
        if client.use_encryption:
            client.certfile = self.remote_certfile
            if self.remote_encryption_key is not None:
                client.encryption_key = self.remote_encryption_key
                if self.remote_encryption_use_aesni is not None:
                    client.encryption_use_aesni = self.remote_encryption_use_aesni
        return client

    def _get_local_client(self) -> Client:
        """
        Returns the local dispatcher client.
        """
        if not self.use_local or not self.config.local:
            raise ValueError("Local dispatcher is not enabled.")
        client = Client()
        client.protocol = self.config.local.protocol
        client.host = self.config.local.host
        client.port = self.config.local.port
        if client.use_encryption:
            client.certfile = self.config.local.certfile
            client.encryption_key = self.config.local.encryption.encryption_key
            client.encryption_use_aesni = self.config.local.encryption.encryption_use_aesni
        return client

    def _get_parameters_from_payload(self, payload: TaskPayload) -> Dict[str, Any]:
        """
        Returns the parameters from a task payload.
        """
        parameters: Optional[Dict[str, Any]] = payload.get("parameters", None)
        if parameters is None:
            return {}
        return parameters

    """Public methods"""

    def call(
        self,
        *args: Any,
        timeout: Optional[Union[int, float]]=None,
        timeout_response: Any=NOTSET,
        target_timeout: Optional[Union[int, float]]=None,
        retries: int=CLIENT_MAX_RETRIES,
        **kwargs: Any
    ) -> Any:
        """
        Calls the taproot dispatcher synchronously.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.__call__(
                *args,
                timeout=timeout,
                timeout_response=timeout_response,
                target_timeout=target_timeout,
                retries=retries,
                **kwargs
            )
        )

    def parallel(
        self,
        *payloads: TaskPayload,
        timeout: Optional[Union[int, float]]=None,
        timeout_response: Any=NOTSET,
        target_timeout: Optional[Union[int, float]]=None,
        retries: int=CLIENT_MAX_RETRIES,
    ) -> List[Any]:
        """
        Calls the taproot dispatcher in parallel synchronously.
        """
        for payload in payloads:
            parameters = payload.get("parameters", None)
            payload["id"] = get_payload_id({} if not isinstance(parameters, dict) else parameters)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            asyncio.gather(*[
                self.__call__(
                    payload["task"],
                    model=payload.get("model", None),
                    timeout=timeout,
                    timeout_response=timeout_response,
                    **self._get_parameters_from_payload(payload)
                )
                for payload in payloads
            ])
        )

    async def get_executor_address(
        self,
        task: str,
        *,
        id: Optional[str]=None,
        model: Optional[str]=None,
        timeout: Optional[Union[int, float]]=None,
        retries: int=CLIENT_MAX_RETRIES,
        **parameters: Any
    ) -> Tuple[str, bool]:
        """
        Gets the executor address for a task.
        """
        task_key = f"{task}:{model}"
        if task_key in self.executor_addresses:
            return self.executor_addresses[task_key], True

        # First we need to get the metadata for the parameters.
        metadata_payload: TaskMetadataPayload = {
            "id": id,
            "task": task,
            "model": model,
            "client_id": self.client_id,
            "parameters": dict([
                (key, get_parameter_metadata(value))
                for key, value in parameters.items()
            ])
        }

        # If there is a local dispatcher, first send the metadata there
        # to get a score for the local machine.
        local_score: Optional[int] = None
        local_client: Optional[Client] = None
        if self.use_local:
            try:
                local_client = self._get_local_client()
                local_score = await local_client(
                    pack_control_message("score", metadata_payload),
                    timeout=0.2, # Short for local dispatcher
                    timeout_response=0,
                )
                if local_score is not None:
                    local_score = int(local_score)
            except Exception as e:
                logger.error(f"Error getting local score: {e}")
                logger.debug(traceback.format_exc())
                local_score = None
                pass

        # Send the metadata payload to the remote overseer if configured.
        executor_address: Optional[str] = None
        if self.use_remote:
            executor_result = await self._get_remote_client().__call__(
                {**metadata_payload, "local_score": local_score},
                timeout=timeout,
                retries=retries,
            )
            executor_address = executor_result["address"]
            id = executor_result["id"]

        if executor_address is None and local_score and local_client:
            # Use the local dispatcher if the overseer doesn't return an executor.
            # Send the prepare payload ourselves to the local dispatcher.
            executor_result = await local_client(
                pack_control_message("prepare", metadata_payload),
                timeout=0.2,
            )
            executor_address = executor_result["address"]
            id = executor_result["id"]
            await asyncio.sleep(0.001) # Sleep to allow the local dispatcher to prepare.

        if not executor_address:
            raise ConnectionError("Could not connect to a local or remote cluster.")

        self.executor_addresses[task_key] = executor_address
        return executor_address, False

    async def __call__(
        self,
        task: str,
        *,
        id: Optional[str]=None,
        model: Optional[str]=None,
        timeout: Optional[Union[int, float]]=None,
        timeout_response: Any=NOTSET,
        target_timeout: Optional[Union[int, float]]=None,
        retries: int=CLIENT_MAX_RETRIES,
        continuation: Optional[Union[TaskPayload, List[TaskPayload]]]=None,
        wait_for_result: bool=True,
        on_intermediate_result: Optional[Callable[[Any], Any]]=None,
        on_tentative_result: Optional[Callable[[Any], Any]]=None,
        **parameters: Any
    ) -> Any:
        """
        Invokes a task on the taproot dispatcher.
        """
        if id is None:
            id = generate_id()

        # Get the executor address for the task.
        executor_address, cached_executor = await self.get_executor_address(
            task,
            id=id,
            model=model,
            timeout=target_timeout,
            retries=retries,
            **parameters,
        )

        # Send the actual task payload to the executor.
        task_payload: TaskPayload = {
            "id": id,
            "client_id": self.client_id,
            "task": task,
            "model": model,
            "parameters": parameters,
            "wait_for_result": wait_for_result,
            "return_metadata": True
        }

        if continuation:
            task_payload["continuation"] = continuation
            task_payload["overseer"] = self.remote_address

        try:
            client = self._get_client_for_address(executor_address)
            result = await client(
                task_payload,
                timeout=timeout,
                timeout_response=timeout_response,
                retries=retries,
            )
        except Exception as e:
            if cached_executor:
                self.executor_addresses.pop(f"{task}:{model}", None)
                return await self.__call__(
                    task,
                    id=id,
                    model=model,
                    timeout=timeout,
                    timeout_response=timeout_response,
                    target_timeout=target_timeout,
                    retries=retries,
                    continuation=continuation,
                    wait_for_result=wait_for_result,
                    on_intermediate_result=on_intermediate_result,
                    on_tentative_result=on_tentative_result,
                    **parameters
                )
            else:
                raise e
        if wait_for_result:
            if continuation:
                continuation_result = result.get("continuation", None)
                if continuation_result is None:
                    logger.warning("Continuation requested but no continuation result found.")
                else:
                    # Recursively query the continuation(s) until they are done.
                    if on_intermediate_result is not None:
                        on_intermediate_result(result["result"])

                    return_first = not isinstance(continuation_result, list)
                    if not isinstance(continuation_result, list):
                        continuation_result = [continuation_result]

                    follow_coro: List[Coroutine[Any, Any, Any]] = []
                    for continuation_payload in continuation_result:
                        follow_coro.append(
                            self._follow_continuation(
                                continuation_payload["address"],
                                continuation_payload["id"],
                                on_intermediate_result=on_intermediate_result,
                                on_tentative_result=on_tentative_result,
                            )
                        )
                    follow_results = await asyncio.gather(*follow_coro)
                    results = [r["result"] for r in follow_results]
                    if return_first:
                        return results[0]
                    return results
            return result["result"]
        return result

    async def _follow_continuation(
        self,
        address: str,
        id: str,
        on_intermediate_result: Optional[Callable[[Any], Any]]=None,
        on_tentative_result: Optional[Callable[[Any], Any]]=None,
    ) -> Any:
        """
        Follows continuation(s) for a task.
        """
        executor = self._get_client_for_address(address)
        result = await executor(
            {
                "id": id,
                "client_id": self.client_id,
                "wait_for_result": True,
                "return_metadata": True
            }
        )
        follow_up_continuation = result.get("continuation", None)
        if follow_up_continuation is not None:
            if on_tentative_result is not None:
                on_tentative_result(result["result"])
            return_first = not isinstance(follow_up_continuation, list)
            if not isinstance(follow_up_continuation, list):
                follow_up_continuation = [follow_up_continuation]
            follow_up_coro: List[Coroutine[Any, Any, Any]] = []
            for continuation_payload in follow_up_continuation:
                follow_up_coro.append(
                    self._follow_continuation(
                        continuation_payload["address"],
                        continuation_payload["id"],
                        on_intermediate_result=on_intermediate_result,
                        on_tentative_result=on_tentative_result,
                    )
                )
            follow_up_results = await asyncio.gather(*follow_up_coro)
            if return_first:
                return follow_up_results[0]
            return follow_up_results
        return result

    @classmethod
    @contextmanager
    def local(
        cls,
        remote: Optional[ClientConfig]=None,
        protocol: Literal["memory", "tcp", "unix", "ws"]="memory",
        host: Optional[str]=None,
        port: Optional[int]=None,
        path: Optional[str]=None,
        max_workers: Optional[int]=None,
        max_idle_time: Optional[float]=None,
        use_multiprocessing: bool=False,
        task_config: Optional[Dict[str, TaskQueueConfigDict]]=None,
        static_executors: List[ExecutorConfigDict]=[],
        task_auto_executors: Optional[Dict[str, int]]=None,
    ) -> Iterator[Tap]:
        """
        Returns the default tap instance.
        """
        from ..server import Dispatcher

        local_executor_config: ExecutorConfigDict = {
            "protocol": "tcp",
            "max_idle_time": max_idle_time,
        }

        if protocol == "tcp" and not host:
            host = DEFAULT_HOST
        elif protocol == "unix" and not path:
            path = find_free_unix_socket()

        if protocol == "tcp" and not port:
            port = find_free_port()
        elif protocol == "memory" and not port:
            port = find_free_memory_port()

        if task_auto_executors is not None:
            for task_key, num_workers in task_auto_executors.items():
                task_name = task_key
                task_model: Optional[str] = None
                if ":" in task_key:
                    task_name, _, task_model = task_key.partition(":")
                for i in range(num_workers):
                    static_executors.append({
                        "protocol": "tcp",
                        "host": host,
                        "port": find_free_port(),
                        "queue_config": {
                            "task": task_name,
                            "model": task_model,
                        }
                    })

        local_config: DispatcherConfigDict = {
            "protocol": protocol,
            "host": host,
            "port": port,
            "path": path,
            "use_multiprocessing": use_multiprocessing,
            "max_workers": max_workers,
            "max_idle_time": max_idle_time,
            "executor_config": local_executor_config,
            "task_config": task_config,
            "static_executor_config": static_executors,
        }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        dispatcher = Dispatcher(local_config) # type:ignore[arg-type]
        task = loop.create_task(dispatcher.run())
        loop.run_until_complete(asyncio.sleep(0.001))

        yield Tap({"local": local_config, "remote": remote})

        dispatcher.graceful_exit()
        if not task.done():
            task.cancel()
        loop.run_until_complete(asyncio.sleep(0.001))
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        loop.close()
