from __future__ import annotations

import ssl
import asyncio
import socket
import struct

from typing import Optional, Any, Tuple, Union, Dict
from typing_extensions import Literal

from time import perf_counter
from websockets.asyncio.client import connect

from ..util import (
    logger,
    get_secret,
    chunk_bytes,
    parse_address,
    format_address,
    pack_and_encode,
    decode_and_unpack,
    pack_control_message,
    unpack_control_message,
    get_top_level_domain_from_url
)
from ..constants import *
from ..encryption import Encryption

class Client(Encryption):
    """
    As base client class for unix or TCP communication.
    """
    _ssl_context: Optional[Union[bool, ssl.SSLContext]]

    """Default properties"""

    @property
    def default_protocol(self) -> Literal["memory", "tcp", "unix", "ws"]:
        """
        Default protocol class for the server.
        """
        return DEFAULT_PROTOCOL # type: ignore

    @property
    def default_host(self) -> str:
        """
        Default host for TCP connections or UNIX sockets.
        """
        return DEFAULT_HOST

    @property
    def default_port(self) -> int:
        """
        Default port for TCP connections or memory-based servers.
        """
        if self.scheme == "ws":
            return 80
        elif self.scheme == "wss":
            return 443
        return DEFAULT_PORT

    @property
    def default_path(self) -> Optional[str]:
        """
        Default path for UNIX sockets or reverse proxies.
        """
        return None

    @property
    def default_use_encryption(self) -> bool:
        """
        Default encryption setting for the client.
        """
        return False

    @property
    def default_certfile(self) -> Optional[str]:
        """
        Default certfile file for the client.
        Only used for websockets when encryption is enabled.
        Also only needed if the server is using a self-signed certfile.
        """
        return None

    @property
    def default_cafile(self) -> Optional[str]:
        """
        Default cafile for the client.
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
            env_var = get_secret(self.control_encryption_var)
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
        elif self.protocol == "ws" and self.use_encryption:
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
            "path": self.path
        })

    @address.setter
    def address(self, value: str) -> None:
        """
        Set the address for the server.
        """
        address = parse_address(value)
        self.scheme = address["scheme"]
        self.path = address["path"]
        if address["host"]:
            self.host = address["host"]
        if address["port"]:
            self.port = address["port"]
        elif self.scheme == "wss":
            self.port = 443
        elif self.scheme == "ws":
            self.port = 80

    @property
    def use_encryption(self) -> bool:
        """
        Encryption setting for the client.
        """
        if not hasattr(self, "_use_encryption"):
            self._use_encryption = self.default_use_encryption
        return self._use_encryption

    @use_encryption.setter
    def use_encryption(self, use_encryption: bool) -> None:
        """
        Set the encryption setting for the client.
        """
        self._use_encryption = use_encryption
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
    def certfile(self) -> Optional[str]:
        """
        certfile file for the client.
        """
        if not hasattr(self, "_certfile"):
            self._certfile = self.default_certfile
        return self._certfile

    @certfile.setter
    def certfile(self, value: Optional[str]) -> None:
        """
        Set the certfile file for the client.
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
        if hasattr(self, "_ssl_context"):
            delattr(self, "_ssl_context")

    @property
    def default_ca(self) -> bool:
        """
        Whether to load default verification locations.
        """
        if not hasattr(self, "_default_ca"):
            self._default_ca = True
        return self._default_ca

    @default_ca.setter
    def default_ca(self, value: bool) -> None:
        """
        Set whether to load default verification locations.
        """
        self._default_ca = value
        if hasattr(self, "_ssl_context"):
            delattr(self, "_ssl_context")

    @property
    def certifi_ca(self) -> bool:
        """
        Whether to load the certifi CA.
        """
        if not hasattr(self, "_certifi_ca"):
            self._certifi_ca = True
        return self._certifi_ca

    @certifi_ca.setter
    def certifi_ca(self, value: bool) -> None:
        """
        Set whether to load the certifi CA.
        """
        self._certifi_ca = value
        if hasattr(self, "_ssl_context"):
            delattr(self, "_ssl_context")

    """Getters only"""

    @property
    def ssl_context(self) -> Optional[Union[bool, ssl.SSLContext]]:
        """
        SSL context for the client.
        """
        if not hasattr(self, "_ssl_context"):
            if self.use_encryption:
                if self.certfile or self.cafile:
                    import certifi
                    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                    if self.default_ca:
                        context.load_default_certs()
                    if self.certfile:
                        context.load_verify_locations(self.certfile)
                    if self.cafile:
                        context.load_verify_locations(self.cafile)
                    if self.certifi_ca:
                        context.load_verify_locations(certifi.where())
                    self._ssl_context = context
                else:
                    self._ssl_context = True
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
            self._control_encryption.encryption_key = self.control_encryption_key # type: ignore[assignment]
            self._control_encryption.encryption_key_length = self.control_encryption_key_length
            self._control_encryption.encryption_use_aesni = self.control_encryption_use_aesni
        return self._control_encryption

    @property
    def websocket_headers(self) -> Dict[str, str]:
        """
        Additional headers for websocket connections.
        """
        headers: Dict[str, str] = {}
        # Check if the target host is a huggingface domain
        tld = get_top_level_domain_from_url(self.address)
        if tld.lower() in HUGGINGFACE_DOMAINS:
            # Look for HF token in environment variables
            token = get_secret("HF_TOKEN")
            if token is not None:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    """Public methods"""

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

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """
        Calls the taproot dispatcher synchronously.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.__call__(*args, **kwargs))

    async def command(
        self,
        command: str,
        data: Any=None,
        **kwargs: Any
    ) -> Any:
        """
        Send a control command to the server.
        """
        return await self(
            self.pack_control_message(command, data),
            **kwargs
        )

    async def __call__(
        self,
        request: Any=None,
        *,
        timeout: Optional[Union[float, int]]=None,
        timeout_response: Any=NOTSET,
        timeout_growth: Optional[float]=None,
        retries: int=CLIENT_MAX_RETRIES,
        retry_delay: Optional[Union[float, int]]=CLIENT_RETRY_DELAY,
        error_response: Any=NOTSET,
        **kwargs: Any
    ) -> Any:
        """
        Execute a request to the server.
        """
        execute_start = perf_counter()
        try:
            if self.protocol == "memory":
                address_label = f"memory[{self.port}]"
                from ..server import get_in_memory_server
                logger.debug(f"Dispatching request directly to in-memory server on port {self.port}.")
                try:
                    server = get_in_memory_server(self.port)
                except ValueError as e:
                    raise ConnectionError(f"Could not connect to in-memory server on port {self.port}. {e}")
                return await server.process(request)
            elif self.protocol == "ws":
                address_label = self.address
                async with connect(
                    self.address,
                    ssl=self.ssl_context,
                    open_timeout=timeout,
                    close_timeout=timeout,
                    ping_timeout=timeout,
                    additional_headers=self.websocket_headers
                ) as websocket:
                    logger.debug(f"Sending message to {self.address} (timeout: {timeout}).")
                    encoded = pack_and_encode(request)
                    encoded_len = struct.pack('!I', len(encoded))
                    encoded = encoded_len + encoded

                    for i, chunk in enumerate(chunk_bytes(encoded, WEBSOCKET_CHUNK_SIZE)):
                        try:
                            await websocket.send(chunk)
                            if i > 0 and i % 2 == 0:
                                await asyncio.sleep(0.001) # Sleep for a millisecond every other chunk to give the server a chance to process
                        except Exception as e:
                            logger.error(f"Error sending message to {address_label}. {e}")
                            raise

                    logger.debug("Message sent, awaiting response.")
                    response = await websocket.recv()
                    if response is None or response == b"":
                        return None
                    if isinstance(response, bytes):
                        response_len = struct.unpack('!I', response[:4])[0]
                        response = response[4:]
                        while len(response) < response_len:
                            chunk = await websocket.recv() # type: ignore[assignment,unused-ignore]
                            assert isinstance(chunk, bytes), "Received non-bytes data from websocket while awaiting additional chunks."
                            response += chunk
                        result = decode_and_unpack(response)
                    else:
                        result = response
                    if isinstance(result, Exception):
                        raise result
                    return result
            else:
                address: Union[str, Tuple[str, int]]
                if self.protocol == "unix":
                    assert self.path is not None, "Path must be set for UNIX sockets."
                    address_label = self.path
                    address = self.path
                    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                else:
                    address_label = f"{self.host}:{self.port}"
                    address = (self.host, self.port)
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                # Connect to the server
                client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                client.setblocking(False)
                logger.debug(f"Connecting to {address_label}.")
                try:
                    await asyncio.get_running_loop().sock_connect(client, address)
                except FileNotFoundError:
                    raise ConnectionError(f"Could not connect to {address_label}.")

                # Prefix the message with its length
                if request is None:
                    message_data = b""
                elif not isinstance(request, bytes):
                    message_data = pack_and_encode(request)
                else:
                    message_data = request

                if self.use_encryption:
                    message_data = self.encrypt(message_data)

                message_len = len(message_data)
                message_length = struct.pack('!I', message_len)

                logger.debug(f"Sending message of length {message_len} to {address_label} ({'encrypted' if self.use_encryption else 'unencrypted'}).")
                await asyncio.get_running_loop().sock_sendall(client, message_length + message_data)
                logger.debug(f"Message sent, awaiting response. Timeout: {timeout} {request=}")
                await asyncio.sleep(.001)

                # Read the echoed message length
                if timeout:
                    length_data = await asyncio.wait_for(
                        asyncio.get_running_loop().sock_recv(client, 4),
                        timeout=timeout
                    )
                else:
                    length_data = await asyncio.get_running_loop().sock_recv(client, 4)

                message_len = int(struct.unpack('!I', length_data)[0])
                logger.debug(f"Received message length {message_len}, reading message data.")

                # Read the echoed message data
                response = bytes()
                while len(response) < message_len:
                    if timeout:
                        packet = await asyncio.wait_for(
                            asyncio.get_running_loop().sock_recv(client, message_len - len(response)),
                            timeout=timeout
                        )
                    else:
                        packet = await asyncio.get_running_loop().sock_recv(client, message_len - len(response))
                    if not packet:
                        break
                    response += packet

                client.close()
                if message_len == 0:
                    return None

                if self.use_encryption:
                    try:
                        response = self.decrypt(response)
                    except Exception as e:
                        logger.error(f"Error decrypting response from {address_label}. {e}")
                try:
                    result = decode_and_unpack(response) # type: ignore[arg-type,unused-ignore]
                except:
                    logger.error(f"Error decoding response from {address_label}.")
                    raise
                if isinstance(result, Exception):
                    raise result
                return result

        except Exception as e:
            if retries > 0:
                logger.debug(f"Error querying {address_label}, retrying up to {retries} more time(s). {type(e).__name__}({e})")
                if self.protocol == "ws" and timeout and retries:
                    # Websockets return immediately, whereas the others will timeout after a time.
                    # To emulate the same behavior, we'll sleep for the remaining time.
                    actual_time = perf_counter() - execute_start
                    remaining_time = timeout - actual_time
                    if remaining_time > 0:
                        await asyncio.sleep(remaining_time)
                elif retry_delay:
                    await asyncio.sleep(retry_delay)
                if timeout_growth and timeout:
                    timeout = timeout * (1 + timeout_growth)
                return await self(
                    request,
                    timeout=timeout,
                    timeout_growth=timeout_growth,
                    timeout_response=timeout_response,
                    error_response=error_response,
                    retries=retries - 1,
                    retry_delay=retry_delay,
                    **kwargs
                )
            if isinstance(e, asyncio.TimeoutError):
                if timeout_response is not NOTSET:
                    logger.debug(f"Timeout querying {address_label}, returning timeout response.")
                    return timeout_response
            if error_response is not NOTSET:
                logger.debug(f"Error querying {address_label}, returning error response.")
                return error_response
            raise # Finally, re-raise the exception
