from __future__ import annotations

import os
import ssl
import time
import struct
import socket
import asyncio
import threading
import traceback
import ipaddress
import websockets

from typing import (
    Any,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from async_lru import alru_cache
from contextlib import contextmanager, nullcontext
from websockets.exceptions import ConnectionClosedOK
from websockets.asyncio.server import ServerConnection

from ..constants import *
from ..payload import *
from ..encryption import Encryption
from ..util import (
    asyncio_run,
    chunk_bytes,
    decode_and_unpack,
    format_address,
    generate_temp_key_and_cert,
    get_default_ip_address,
    is_control_message,
    logger,
    pack_and_encode,
    pack_control_message,
    parse_address,
    timed_lru_cache,
    unpack_control_message,
    MachineCapability,
)

if TYPE_CHECKING:
    from ..client import Client

class Server(Encryption):
    """
    Main class for running a server, either locally or over a network.
    """
    _ssl_context: Optional[ssl.SSLContext]
    _start_time: float
    _num_requests: int
    _shutdown_key: str

    @property
    def default_protocol(self) -> Literal["memory", "tcp", "unix", "ws"]:
        """
        Default protocol for the server.
        """
        return DEFAULT_PROTOCOL # type: ignore[return-value]

    @property
    def default_host(self) -> str:
        """
        Default host for the server.
        """
        return DEFAULT_HOST

    @property
    def default_port(self) -> int:
        """
        Default port for the server.
        """
        if self.scheme == "ws":
            return 80
        elif self.scheme == "wss":
            return 443
        return DEFAULT_PORT

    @property
    def default_path(self) -> Optional[str]:
        """
        Default path for the server.
        Path applies to UNIX sockets or can be relevant when behind a reverse proxy.
        """
        return None

    @property
    def default_use_encryption(self) -> bool:
        """
        Default encryption setting for the server.
        """
        return False

    @property
    def default_allow_list(self) -> Optional[List[str]]:
        """
        Default IP allow_list for the server for TCP and WS.
        Default is to allow all connections.
        """
        return None

    @property
    def default_control_list(self) -> Optional[List[str]]:
        """
        Default IP control_list for the server.
        Default is to only allow connections from localhost.
        """
        return [
            "127.0.0.1/24",
            f"{get_default_ip_address()}/32",
        ]

    @property
    def default_max_idle_time(self) -> Optional[float]:
        """
        Default maximum idle time for the server.
        """
        return None

    @property
    def default_keyfile(self) -> Optional[str]:
        """
        Default keyfile for the server.
        Only used when using websockets with encryption.
        """
        return None

    @property
    def default_certfile(self) -> Optional[str]:
        """
        Default certfile for the server.
        Only used when using websockets with encryption.
        """
        return None

    @property
    def default_cafile(self) -> Optional[str]:
        """
        Default cafile for the server.
        Only used when using websockets with encryption.
        """
        return None

    @property
    def default_use_control_encryption(self) -> bool:
        """
        Default encryption setting for control requests.
        """
        return False

    @property
    def default_control_encryption_key(self) -> Optional[bytes]:
        """
        Default encryption key for control requests.
        """
        if self.control_encryption_var is not None:
            env_var = os.getenv(self.control_encryption_var, None)
            if env_var is not None:
                return env_var.encode("utf-8")
        return None

    @property
    def default_control_encryption_var(self) -> Optional[str]:
        """
        Default environment variable for the encryption key for control requests.
        """
        return None

    @property
    def default_control_encryption_key_length(self) -> int:
        """
        Default encryption key length for control requests.
        """
        return 32

    @property
    def default_control_encryption_use_aesni(self) -> bool:
        """
        Default encryption setting for control requests.
        """
        return True

    """Getter/setter properties"""

    @property
    def protocol(self) -> Literal["memory", "tcp", "unix", "ws"]:
        """
        Protocol class for the server.
        """
        if not hasattr(self, "_protocol"):
            self._protocol = self.default_protocol
        return self._protocol

    @protocol.setter
    def protocol(self, value: Literal["memory", "tcp", "unix", "ws"]) -> None:
        """
        Set the protocol class for the server.
        """
        self._protocol = value

    @property
    def scheme(self) -> Literal["memory", "tcp", "tcps", "unix", "ws", "wss"]:
        """
        Scheme for the server.
        """
        if self.protocol == "tcp" and self.use_encryption:
            return "tcps"
        if self.protocol == "ws" and self.use_encryption:
            return "wss"
        return self.protocol

    @scheme.setter
    def scheme(self, value: Literal["memory", "tcp", "tcps", "unix", "ws", "wss"]) -> None:
        """
        Set the scheme for the server.
        """
        if value == "tcps":
            self.protocol = "tcp"
            self.use_encryption = True
        elif value == "wss":
            self.protocol = "ws"
            self.use_encryption = True
        else:
            self.protocol = value
            self.use_encryption = False

    @property
    def host(self) -> str:
        """
        Host for WS/TCP connections or UNIX sockets.
        """
        if not hasattr(self, "_host"):
            self._host = self.default_host
        return self._host

    @host.setter
    def host(self, value: str) -> None:
        """
        Set the host for WS/TCP connections or UNIX sockets.
        """
        self._host = value

    @property
    def port(self) -> int:
        """
        Port for WS/TCP connections or memory-based servers.
        """
        if not hasattr(self, "_port"):
            self._port = self.default_port
        return self._port

    @port.setter
    def port(self, value: int) -> None:
        """
        Set the port for WS/TCP connections or memory-based servers.
        """
        self._port = value

    @property
    def path(self) -> Optional[str]:
        """
        Path for UNIX sockets or reverse proxies.
        """
        if not hasattr(self, "_path"):
            self._path = self.default_path
        return self._path

    @path.setter
    def path(self, value: Optional[str]) -> None:
        """
        Set the path for UNIX sockets or reverse proxies.
        """
        self._path = value

    @property
    def address(self) -> str:
        """
        Address for the server.
        """
        return format_address({
            "scheme": self.scheme,
            "host": self.host,
            "port": self.port,
            "path": self.path,
        })

    @address.setter
    def address(self, value: str) -> None:
        """
        Set the address for the server.
        """
        address = parse_address(value)
        self.scheme = address["scheme"]
        self.path = address["path"]
        if address["port"]:
            self.port = address["port"]
        elif self.scheme == "wss":
            self.port = 443
        elif self.scheme == "ws":
            self.port = 80

    @property
    def external_address(self) -> str:
        """
        External address for the server.
        """
        if not hasattr(self, "_external_address"):
            return self.address
        return self._external_address

    @external_address.setter
    def external_address(self, value: str) -> None:
        """
        Set the external address for the server.
        """
        self._external_address = value

    @property
    def allow_list(self) -> Optional[List[str]]:
        """
        IP allow_list for TCP.
        """
        if not hasattr(self, "_allow_list"):
            self._allow_list = self.default_allow_list
        return self._allow_list

    @allow_list.setter
    def allow_list(self, value: Optional[List[str]]) -> None:
        """
        Set the IP allow_list for TCP.
        """
        self._allow_list = value

    @property
    def control_list(self) -> Optional[List[str]]:
        """
        IP allow_list for control requests.
        """
        if not hasattr(self, "_control_list"):
            self._control_list = self.default_control_list
        return self._control_list

    @control_list.setter
    def control_list(self, value: Optional[List[str]]) -> None:
        """
        Set the IP allow_list for control requests.
        """
        self._control_list = value

    @property
    def max_idle_time(self) -> Optional[float]:
        """
        Maximum idle time for a server before it is shut down.
        """
        if not hasattr(self, "_max_idle_time"):
            self._max_idle_time = self.default_max_idle_time
        return self._max_idle_time

    @max_idle_time.setter
    def max_idle_time(self, value: Optional[float]) -> None:
        """
        Set the maximum idle time for a server.
        """
        self._max_idle_time = value

    @property
    def processing(self) -> bool:
        """
        Whether the server is processing a request.
        """
        if not hasattr(self, "_processing"):
            self._processing = False
        return self._processing

    @property
    def manual_exit(self) -> bool:
        """
        Whether the server has received an exit request.
        """
        if not hasattr(self, "_manual_exit"):
            self._manual_exit = False
        return self._manual_exit

    @manual_exit.setter
    def manual_exit(self, value: bool) -> None:
        """
        Set whether the server has received an exit request.
        """
        self._manual_exit = value

    @property
    def use_encryption(self) -> bool:
        """
        Whether to use encryption for WS/TCP connections.
        """
        if not hasattr(self, "_use_encryption"):
            self._use_encryption = self.default_use_encryption
        return self._use_encryption

    @use_encryption.setter
    def use_encryption(self, value: bool) -> None:
        """
        Set whether to use encryption.
        """
        self._use_encryption = value
        if hasattr(self, "_ssl_context"):
            delattr(self, "_ssl_context")

    @property
    def use_control_encryption(self) -> bool:
        """
        Whether to use encryption for control requests.
        """
        if not hasattr(self, "_use_control_encryption"):
            self._use_control_encryption = self.default_use_control_encryption
        return self._use_control_encryption

    @use_control_encryption.setter
    def use_control_encryption(self, value: bool) -> None:
        """
        Set whether to use encryption for control requests.
        """
        self._use_control_encryption = value

    @property
    def control_encryption_key(self) -> Optional[bytes]:
        """
        Key for control request encryption.
        """
        if not hasattr(self, "_control_encryption_key"):
            self._control_encryption_key = self.default_control_encryption_key
        return self._control_encryption_key

    @control_encryption_key.setter
    def control_encryption_key(self, value: Optional[Union[str, bytes]]) -> None:
        """
        Set the key for control request encryption.
        """
        if isinstance(value, str):
            value = value.encode("utf-8")
        self._control_encryption_key = value
        if hasattr(self, "_control_encryption"):
            delattr(self, "_control_encryption")

    @property
    def control_encryption_var(self) -> Optional[str]:
        """
        Environment variable for the control encryption key.
        """
        if not hasattr(self, "_control_encryption_var"):
            self._control_encryption_var = self.default_control_encryption_var
        return self._control_encryption_var

    @control_encryption_var.setter
    def control_encryption_var(self, value: Optional[str]) -> None:
        """
        Set the environment variable for the control encryption key.
        """
        self._control_encryption_var = value
        if hasattr(self, "_control_encryption"):
            delattr(self, "_control_encryption")

    @property
    def control_encryption_key_length(self) -> int:
        """
        Key length for control request encryption.
        """
        if not hasattr(self, "_control_encryption_key_length"):
            self._control_encryption_key_length = self.default_control_encryption_key_length
        return self._control_encryption_key_length

    @control_encryption_key_length.setter
    def control_encryption_key_length(self, value: int) -> None:
        """
        Set the key length for control request encryption.
        """
        self._control_encryption_key_length = value
        if hasattr(self, "_control_encryption"):
            delattr(self, "_control_encryption")

    @property
    def control_encryption_use_aesni(self) -> bool:
        """
        Whether to use AESNI for control request encryption.
        """
        if not hasattr(self, "_control_encryption_use_aesni"):
            self._control_encryption_use_aesni = self.default_control_encryption_use_aesni
        return self._control_encryption_use_aesni

    @control_encryption_use_aesni.setter
    def control_encryption_use_aesni(self, value: bool) -> None:
        """
        Set whether to use AESNI for control request encryption.
        """
        self._control_encryption_use_aesni = value
        if hasattr(self, "_control_encryption"):
            delattr(self, "_control_encryption")

    @property
    def keyfile(self) -> Optional[str]:
        """
        Keyfile for encryption.
        Only used when using websockets with encryption.
        """
        if not hasattr(self, "_keyfile"):
            self._keyfile = self.default_keyfile
        return self._keyfile

    @keyfile.setter
    def keyfile(self, value: Optional[str]) -> None:
        """
        Set the keyfile for encryption.
        """
        self._keyfile = value
        if hasattr(self, "_ssl_context"):
            delattr(self, "_ssl_context")

    @property
    def certfile(self) -> Optional[str]:
        """
        Certfile for encryption.
        Only used when using websockets with encryption.
        """
        if not hasattr(self, "_certfile"):
            self._certfile = self.default_certfile
        return self._certfile

    @certfile.setter
    def certfile(self, value: Optional[str]) -> None:
        """
        Set the certfile for encryption.
        """
        self._certfile = value
        if hasattr(self, "_ssl_context"):
            delattr(self, "_ssl_context")

    @property
    def cafile(self) -> Optional[str]:
        """
        CA file for encryption.
        Only used when using websockets with encryption.
        """
        if not hasattr(self, "_cafile"):
            self._cafile = self.default_cafile
        return self._cafile

    @cafile.setter
    def cafile(self, value: Optional[str]) -> None:
        """
        Set the CA file for encryption.
        """
        self._cafile = value

    """Getters Only"""

    @property
    def lock(self) -> threading.Lock:
        """
        Lock for the server.
        """
        if not hasattr(self, "_lock"):
            self._lock = threading.Lock()
        return self._lock

    @property
    def ssl_context(self) -> Optional[ssl.SSLContext]:
        """
        SSL context for the server.
        """
        if not hasattr(self, "_ssl_context"):
            if self.use_encryption:
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                if self.certfile is None or self.keyfile is None:
                    logger.warning(f"Keyfile and certfile not set. Generating temporary self-signed certificates for {type(self).__name__}. This is not secure and should not be used in production.")
                    self.keyfile, self.certfile = generate_temp_key_and_cert()
                assert os.path.exists(self.certfile), f"certfile {self.certfile} does not exist."
                assert os.path.exists(self.keyfile), f"keyfile {self.keyfile} does not exist."
                ssl_context.load_cert_chain(certfile=self.certfile, keyfile=self.keyfile)
                self._ssl_context = ssl_context
            else:
                self._ssl_context = None
        return self._ssl_context

    @property
    def control_encryption(self) -> Optional[Encryption]:
        """
        Encryption for control requests.
        """
        if not self.use_control_encryption:
            return None
        if not hasattr(self, "_control_encryption"):
            self._control_encryption = Encryption()
            if self.control_encryption_key is not None:
                self._control_encryption.encryption_key = self.control_encryption_key
            else:
                self._control_encryption.encryption_key_length = self.control_encryption_key_length
            self._control_encryption.encryption_use_aesni = self.control_encryption_use_aesni
        return self._control_encryption

    """Internal methods"""

    def _is_ip_allowed(self, ip: str) -> bool:
        """
        Check if an IP address is allowed to connect to the server.
        """
        if self.allow_list is None:
            return True
        ip_addr = ipaddress.ip_address(ip)
        for subnet in self.allow_list:
            if ip_addr in ipaddress.ip_network(subnet, False):
                return True
        return False

    def _is_ip_allowed_control(self, ip: str) -> bool:
        """
        Check if an IP address is allowed to send control requests to the server.
        The `_is_ip_allowed` method should have already been called, this is a secondary
        permission check.
        """
        if self.control_list is None:
            return True
        ip_addr = ipaddress.ip_address(ip)
        for subnet in self.control_list:
            if ip_addr in ipaddress.ip_network(subnet, False):
                return True
        return False

    def _get_last_request_time(self) -> float:
        """
        Time of the last request.
        """
        if not hasattr(self, "_last_request_time"):
            self._last_request_time = 0.0
        return self._last_request_time

    def _reset_last_request_time(self) -> None:
        """
        Reset the last request time.
        """
        self._last_request_time = time.perf_counter()

    def _get_time_since_last_request(self) -> float:
        """
        Get the time since the last request.
        """
        return time.perf_counter() - self._get_last_request_time()

    def _is_idle_timeout(self) -> bool:
        """
        Check if the server has reached the idle timeout.
        """
        if self.max_idle_time is None:
            return False
        return self._get_time_since_last_request() > self.max_idle_time

    async def _handle_control_request(self, request: str) -> Any:
        """
        Handle a control request.
        """
        try:
            command, data = self.unpack_control_message(request)
        except Exception as e:
            logger.error(f"Error unpacking control message: {e}")
            raise ValueError("Invalid control message.")
        if command == "keepalive":
            return None
        elif command == "capability":
            return self.get_reported_capability()
        elif command == "status":
            return await self.status(data)
        elif command == "exit":
            self.shutdown()
            return self._shutdown_key
        return await self.command(command, data)

    async def _handle_request(self, request: Any) -> Any:
        """
        Handle a request.
        """
        try:
            self._processing = True
            response = await self.handle(request)
            return response
        finally:
            self._processing = False

    async def _increment_active_requests(self) -> None:
        """
        Increment the number of active requests.
        """
        self._reset_last_request_time()
        with self.lock:
            self._num_requests += 1
            if not hasattr(self, "_active_requests"):
                self._active_requests = 0
            self._active_requests += 1

    async def _decrement_active_requests(self) -> None:
        """
        Decrement the number of active requests.
        """
        with self.lock:
            if not hasattr(self, "_active_requests"):
                self._active_requests = 1
            self._active_requests -= 1

    def _has_active_requests(self) -> bool:
        """
        Check if there are active requests.
        """
        return getattr(self, "_active_requests", 0) > 0

    """Extensible methods for implementations"""

    @contextmanager
    def context(self) -> Iterator[None]:
        """
        Runtime context for the server.
        """
        self._shutdown_key = os.urandom(32).hex()
        self._start_time = time.perf_counter()
        self._num_requests = 0
        if self.protocol == "memory":
            from .memory import set_in_memory_server
            set_in_memory_server(self.port, self)
        yield
        if self.protocol == "memory":
            from .memory import unset_in_memory_server
            unset_in_memory_server(self.port)

    async def handle(self, request: Any) -> Any:
        """
        Handle a request.
        The default implementation simply returns the request.
        """
        return request

    async def command(self, request: str, data: Any=None) -> Any:
        """
        Handle a command.
        The default implementation simply returns the request.
        """
        return request

    async def status(self, data: Any=None) -> ServerStatusPayload:
        """
        Get the status of the server.
        """
        return {
            "active_requests": getattr(self, "_active_requests", 0),
            "processing": self.processing,
            "uptime": time.perf_counter() - self._start_time,
            "num_requests": self._num_requests,
        }

    def shutdown(self) -> None:
        """
        Shutdown the server. Base implementation does nothing.
        """
        logger.info(f"Shutting down {type(self).__name__} on {self.address}.")
        self.manual_exit = True
        return

    """Public methods"""

    @timed_lru_cache(ttl=1.0) # 1 second cache
    def get_capability(self) -> MachineCapability:
        """
        Get the capability of the machine.
        """
        return MachineCapability.get_capability(fail_on_gpu_error=False)

    def pack_control_message(self, message: str, data: Any=None) -> str:
        """
        Pack a control message.
        """
        return pack_control_message(
            message,
            data,
            encryption=self.control_encryption
        )

    def unpack_control_message(self, message: str) -> Tuple[str, Any]:
        """
        Unpack a control message.
        """
        return unpack_control_message(
            message,
            encryption=self.control_encryption
        )

    def get_reported_capability(self) -> CapabilityPayload:
        """
        Get the capability of the machine.
        """
        capability = self.get_capability()
        response: CapabilityPayload = {
            "gpu_memory_bandwidth_gb_s": 0.0,
            "gpu_half_float_performance_gflop_s": 0.0,
            "gpu_single_float_performance_gflop_s": 0.0,
            "gpu_double_float_performance_gflop_s": 0.0,
        }
        for gpu in capability.gpus:
            response["gpu_memory_bandwidth_gb_s"] = max(gpu.specification.memory_bandwidth_gb_s, response["gpu_memory_bandwidth_gb_s"])
            response["gpu_half_float_performance_gflop_s"] = max(gpu.specification.half_float_performance_gflop_s, response["gpu_half_float_performance_gflop_s"])
            response["gpu_single_float_performance_gflop_s"] = max(gpu.specification.single_float_performance_gflop_s, response["gpu_single_float_performance_gflop_s"])
            response["gpu_double_float_performance_gflop_s"] = max(gpu.specification.double_float_performance_gflop_s, response["gpu_double_float_performance_gflop_s"])
        return response

    async def process(self, request: Any) -> Any:
        """
        Handle a request for memory-based servers.
        """
        await self._increment_active_requests()
        try:
            if is_control_message(request):
                return await self._handle_control_request(request)
            else:
                return await self._handle_request(request)
        finally:
            await self._decrement_active_requests()

    async def run(self) -> None:
        """
        Run the server.
        """
        self.manual_exit = False
        manual_exit = False

        logger.info(f"Beginning main server loop on {self.address} for {type(self).__name__}")
        server_context = nullcontext()
        loop = asyncio.get_running_loop()
        server = None

        if self.protocol == "ws":
            # Define scoped handler
            async def handle_websocket(
                websocket: ServerConnection,
                path: Optional[str]=None
            ) -> None:
                """
                Handle a websocket connection.

                :param websocket: The websocket connection.
                :param path: The path of the websocket connection.
                             This will be set in websockets<14, but not in websockets>=14.
                             Keep it as optional for compatibility.
                             If, in the future, we need to use the path variable, we can
                             find it in websocket.request.path in websockets>=14.
                """
                logger.debug(f"Handling websocket connection {websocket}")
                if not self._is_ip_allowed(websocket.remote_address[0]):
                    logger.error(f"Connection from {websocket.remote_address} not allowed.")
                    await websocket.close()
                    return
                request_buffer: bytes = b""
                request_len: int = 0
                try:
                    await self._increment_active_requests()
                    async for message in websocket:
                        request: Any = None
                        response: Any = None
                        if message and isinstance(message, bytes):
                            if not request_buffer:
                                request_len = struct.unpack('!I', message[:4])[0]
                                message = message[4:]
                            request_buffer += message
                            if len(request_buffer) >= request_len:
                                try:
                                    request = decode_and_unpack(request_buffer[:request_len])
                                except Exception as e:
                                    logger.error(f"{type(self).__name__}: Error decoding data: {e}")
                                    logger.debug(traceback.format_exc())
                                    response = e
                                request_buffer = b""
                                request_len = 0
                        if is_control_message(request):
                            if not self._is_ip_allowed_control(websocket.remote_address[0]):
                                # Respond with exception
                                response = PermissionError(f"Control request not allowed from {websocket.remote_address[0]}.")
                            else:
                                try:
                                    response = await self._handle_control_request(request)
                                    if response == self._shutdown_key:
                                        manual_exit = True
                                except Exception as e:
                                    response = e
                        elif request is not None:
                            logger.debug(f"Handling request, payload is of type {type(request)}")
                            try:
                                response = await self._handle_request(request)
                                self._reset_last_request_time()
                            except Exception as e:
                                logger.error(f"{type(self).__name__}: Error handling request: {e}")
                                logger.debug(traceback.format_exc())
                                response = e
                        if request is not None:
                            response = pack_and_encode(response)
                            num_bytes = len(response)
                            response_len = struct.pack('!I', num_bytes)
                            response = response_len + response
                            num_chunks = int(num_bytes // WEBSOCKET_CHUNK_SIZE) + 1
                            logger.debug(f"Serving response of length {num_bytes} in {num_chunks} chunk(s).")
                            for chunk in chunk_bytes(response, WEBSOCKET_CHUNK_SIZE):
                                try:
                                    await websocket.send(chunk)
                                except ConnectionClosedOK:
                                    continue
                                self._reset_last_request_time()
                finally:
                    await self._decrement_active_requests()

            # Define server context to enter
            server_context = websockets.serve( # type: ignore[assignment]
                handle_websocket,
                self.host,
                self.port,
                ssl=self.ssl_context
            )
        elif not self.protocol == "memory":
            if self.protocol == "unix":
                assert self.path is not None, "Path must be set for UNIX sockets."

            if self.protocol == "unix" and os.path.exists(self.path): # type: ignore[arg-type]
                os.remove(self.path) # type: ignore[arg-type]

            if self.protocol == "unix":
                server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    server.bind(self.path) # type: ignore[arg-type]
                except Exception as e:
                    raise RuntimeError(f"Error binding to {self.path}") from e
            else:
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    server.bind((self.host, self.port))
                except Exception as e:
                    raise RuntimeError(f"Error binding to {self.host}:{self.port}") from e

            server.listen()
            server.setblocking(False)

            async def handle_client(client_socket: socket.socket) -> None:
                """
                Handle a client connection.
                """
                nonlocal manual_exit
                logger.debug(f"Handling client connection {client_socket}")
                await self._increment_active_requests()
                try:
                    with client_socket:
                        response: bytes = b""
                        while True:
                            try:
                                # Read the message length
                                try:
                                    length_data = await asyncio.wait_for(
                                        loop.sock_recv(client_socket, 4),
                                        timeout=SERVER_POLLING_INTERVAL
                                    )
                                except asyncio.TimeoutError:
                                    length_data = b""

                                if not length_data:
                                    break

                                message_length = struct.unpack('!I', length_data)[0]
                                logger.debug(f"Received message length {message_length}, reading data.")

                                # Read the message data
                                data = b''
                                while len(data) < message_length:
                                    try:
                                        packet = await asyncio.wait_for(
                                            loop.sock_recv(
                                                client_socket,
                                                message_length-len(data)
                                            ),
                                            timeout=SERVER_POLLING_INTERVAL
                                        )
                                    except asyncio.TimeoutError:
                                        packet = b""
                                    if not packet:
                                        break
                                    data += packet

                                # Decode the data and execute the task
                                if self.use_encryption and data:
                                    try:
                                        data = self.decrypt(data)
                                    except Exception as e:
                                        logger.error(f"{type(self).__name__}: Error decrypting data: {e}")

                                if data:
                                    try:
                                        request = decode_and_unpack(data)
                                    except Exception as e:
                                        logger.error(f"{type(self).__name__}: Error unpickling data: {e}")
                                        request = None
                                else:
                                    request = None

                                # Check if the request is a control request while we're in scope
                                if is_control_message(request):
                                    logger.debug(f"Handling control request: {request}")
                                    peername = client_socket.getpeername()
                                    if self.protocol != "unix" and not self._is_ip_allowed_control(peername[0]):
                                        # Respond with exception
                                        result = PermissionError("Control request not allowed from {peername[0]}.")
                                    else:
                                        result = await self._handle_control_request(request)
                                        if isinstance(result, str) and result == self._shutdown_key:
                                            manual_exit = True
                                else:
                                    logger.debug(f"Handling request, payload is of type {type(request)}")
                                    logger.debug(f"Request: {request}")
                                    result = await self._handle_request(request)

                                response = pack_and_encode(result)
                            except Exception as e:
                                if getattr(e, "errno", None) == 104: # Connection reset by peer
                                    break
                                logger.error(f"Error handling request: {e}")
                                logger.debug(traceback.format_exc())
                                response = pack_and_encode(e)
                            try:
                                if self.use_encryption and response:
                                    response = self.encrypt(response)
                                response_length = struct.pack('!I', len(response))
                                logger.debug(f"Sending response of length {len(response)}")
                                await loop.sock_sendall(client_socket, response_length + response)
                            except Exception as e:
                                if getattr(e, "errno", None) == 32: # Broken pipe
                                    break
                                logger.error(f"Error sending response: {e}")
                                break
                finally:
                    await self._decrement_active_requests()
        try:
            with self.context():
                async with server_context:
                    self._reset_last_request_time()
                    while True:
                        if manual_exit or self.manual_exit:
                            logger.info(f"{type(self).__name__} listening on {self.address} received exit request.")
                            break
                        try:
                            if self.protocol in ["memory", "ws"]:
                                # We sleep manually for these two protocols
                                try:
                                    await asyncio.sleep(SERVER_POLLING_INTERVAL)
                                except:
                                    raise asyncio.CancelledError
                            else:
                                if not server:
                                    raise RuntimeError(f"{self.protocol} server failed to initialize, check logs.")
                                # We'll use the loops sock_accept and wait_for for tcp and unix
                                client_socket, addr = await asyncio.wait_for(
                                    loop.sock_accept(server),
                                    timeout=SERVER_POLLING_INTERVAL
                                )
                                if self.protocol != "unix":
                                    client_ip = addr[0]
                                    if not self._is_ip_allowed(client_ip):
                                        client_socket.close()
                                        continue
                                logger.debug(f"Accepted connection from {client_socket}")
                                loop.create_task(handle_client(client_socket))
                                self._reset_last_request_time()
                        except asyncio.TimeoutError:
                            pass
                        except asyncio.CancelledError:
                            logger.info(f"{type(self).__name__} listening on {self.address} was cancelled.")
                            return
                        except RuntimeError as e:
                            logger.error(f"{type(self).__name__}: RuntimeError in server loop: {e}")
                            break
                        except Exception as e:
                            logger.error(f"{type(self).__name__}: Error in server loop: {e}")
                            pass
                        if self._has_active_requests():
                            self._reset_last_request_time()
                        elif self._is_idle_timeout():
                            logger.info(f"{type(self).__name__} listening on {self.address} reached idle timeout.")
                            return
        except Exception as e:
            logger.error(f"{type(self).__name__}: Error in server loop: {e}")
            logger.error(traceback.format_exc())
            return
        finally:
            try:
                if server is not None:
                    server.close()
            except Exception as e:
                pass
            finally:
                if self.protocol == "unix" and self.path is not None and os.path.exists(self.path):
                    try:
                        os.remove(self.path)
                    except:
                        pass
                self.shutdown()

    def serve(self) -> None:
        """
        Serve the server synchronously.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.run())

    def get_client(self) -> Client:
        """
        Get a client for the server.
        """
        return self.get_client_for_address(self.address)

    def get_client_for_address(self, address: str) -> Client:
        """
        Gets a client for a specific address.
        """
        from ..client import Client
        client = Client()
        client.address = address
        if client.use_encryption:
            client.cafile = self.cafile
            client.certfile = self.certfile
            client.encryption_key = self.encryption_key
            client.encryption_use_aesni = self.encryption_use_aesni
        if self.use_control_encryption:
            client.use_control_encryption = True
            client.control_encryption_key = self.control_encryption_key
            client.control_encryption_use_aesni = self.control_encryption_use_aesni
        return client

    async def exit(self, timeout: Optional[float]=0.05) -> None:
        """
        Exit the server.
        """
        client = self.get_client()
        logger.debug(f"Sending exit request to {self.address}")
        try:
            await client(
                self.pack_control_message("exit"),
                timeout=timeout
            )
        except ConnectionClosedOK:
            pass
        logger.debug("Exit request was successful.")

    def graceful_exit(self, timeout: Optional[float]=0.05) -> bool:
        """
        Gracefully exit the server.
        This is more complicated than it seems due to multi-processing,
        the most robust way is to make a client and then use it to request
        an exit.
        """
        try:
            asyncio.get_running_loop().run_until_complete(
                self.exit(timeout=timeout)
            )
            return True
        except:
            return False

    def wait_for_connectivity(
        self,
        timeout: Optional[float]=0.1,
        timeout_growth: Optional[float]=0.05,
        retries: int=15,
    ) -> None:
        """
        Wait for the server to be running and can be connected to.
        """
        assert asyncio_run(
            self.assert_connectivity(
                timeout=timeout,
                timeout_growth=timeout_growth,
                retries=retries,
            )
        ) is None

    async def assert_connectivity(
        self,
        timeout: Optional[float]=0.1,
        timeout_growth: Optional[float]=0.5,
        retries: int=15,
    ) -> None:
        """
        Assert that the server is running and can be connected to.
        """
        client = self.get_client()
        assert await client(
            self.pack_control_message("keepalive"),
            timeout=timeout,
            timeout_growth=timeout_growth,
            retries=retries,
        ) is None

    @alru_cache(ttl=0.5) # 1/2 second cache
    async def get_status(
        self,
        timeout: Optional[float]=0.1,
        retries: int=0,
        data: Any=None,
    ) -> ServerStatusPayload:
        """
        Returns the status of the executor using the client.
        """
        request_payload = self.pack_control_message("status", data)
        default_result: ServerStatusPayload = {
            "active_requests": 0,
            "processing": False,
            "uptime": 0.0,
            "num_requests": 0,
        }
        try:
            return await self.get_client()( # type: ignore[no-any-return]
                request_payload,
                timeout=timeout,
                timeout_result=default_result,
                retries=retries
            )
        except Exception as e:
            logger.warning(f"Failed to get status: {type(e).__name__}({e})")
            return default_result
