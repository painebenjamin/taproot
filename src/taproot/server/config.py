from __future__ import annotations

from typing import List, Literal, Optional, Union

from ..config import *
from .base import Server

__all__ = ["ConfigServer"]

class ConfigServer(Server, ConfigMixin):
    """
    Server class that uses a configuration file to set default values.
    """
    config_class = ServerConfig
    
    @property
    def default_use_encryption(self) -> bool:
        """
        Override default encryption setting with configured encryption setting.
        """
        return (self.protocol in ["tcp", "unix", "memory"] and self.config.encryption is not None) or \
               (self.protocol == "ws" and self.keyfile is not None and self.certfile is not None)

    @property
    def default_use_control_encryption(self) -> bool:
        """
        Override default control encryption setting with configured control encryption setting.
        """
        return self.protocol in ["tcp", "unix", "ws"] and \
               self.config.control_encryption is not None and \
               self.control_encryption_key is not None

    @property
    def protocol(self) -> Literal["tcp", "unix", "memory", "ws"]:
        """
        Return configured protocol if set, otherwise use default.
        """
        if not self.config.protocol:
            return super().protocol
        return str(self.config.protocol) # type: ignore

    @protocol.setter
    def protocol(self, value: Literal["tcp", "unix", "memory", "ws"]) -> None:
        """
        Set the protocol to use.
        """
        self.config.protocol = value

    @property
    def host(self) -> str:
        """
        Return configured host if set, otherwise use default.
        """
        if not self.config.host:
            return super().host
        return str(self.config.host)

    @host.setter
    def host(self, value: str) -> None:
        """
        Set the host to use.
        """
        self.config.host = value

    @property
    def port(self) -> int:
        """
        Return configured port if set, otherwise use default.
        """
        if not self.config.port:
            return super().port
        return int(self.config.port)

    @port.setter
    def port(self, value: int) -> None:
        """
        Set the port to use.
        """
        self.config.port = value

    @property
    def path(self) -> Optional[str]:
        """
        Return configured path if set, otherwise use default.
        """
        if not self.config.path:
            return super().path
        return str(self.config.path)

    @path.setter
    def path(self, value: Optional[str]) -> None:
        """
        Set the path to use.
        """
        self.config.path = value

    @property
    def external_address(self) -> str:
        """
        The external address for the server.
        """
        if not self.config.external_address:
            return super().external_address
        if self.config.external_address is None:
            return self.address
        return str(self.config.external_address)

    @external_address.setter
    def external_address(self, value: Optional[str]) -> None:
        """
        Set the external address to use.
        """
        self.config.external_address = value

    @property
    def encryption_key_length(self) -> int:
        """
        Return configured encryption key length if set, otherwise use default.
        Only used when there is no encryption key configured.
        """
        if not self.config.encryption or not self.config.encryption.encryption_key_length:
            return super().encryption_key_length
        return int(self.config.encryption.encryption_key_length)

    @encryption_key_length.setter
    def encryption_key_length(self, value: int) -> None:
        """
        Set the encryption key length to use.
        """
        if not self.config.encryption:
            self.config.encryption = EncryptionConfig()
        self.config.encryption.encryption_key_length = value

    @property
    def encryption_key(self) -> bytes:
        """
        Return configured encryption key if set, otherwise use default.
        """
        if not self.config.encryption or not self.config.encryption.encryption_key:
            return super().encryption_key
        return str(self.config.encryption.encryption_key).encode("utf-8")

    @encryption_key.setter
    def encryption_key(self, value: Union[bytes, str]) -> None:
        """
        Set the encryption key to use.
        """
        if not self.config.encryption:
            self.config.encryption = EncryptionConfig()
        self.config.encryption.encryption_key = value.decode("utf-8") if isinstance(value, bytes) else value

    @property
    def encryption_use_aesni(self) -> bool:
        """
        Return configured encryption use AES-NI if set, otherwise use default.
        """
        if not self.config.encryption:
            return super().encryption_use_aesni
        return bool(self.config.encryption.encryption_use_aesni)

    @encryption_use_aesni.setter
    def encryption_use_aesni(self, value: bool) -> None:
        """
        Set whether to use AES-NI for encryption.
        """
        if not self.config.encryption:
            self.config.encryption = EncryptionConfig()
        self.config.encryption.encryption_use_aesni = value

    @property
    def control_encryption_key_length(self) -> int:
        """
        Return configured control encryption key length if set, otherwise use default.
        Only used when there is no control encryption key configured.
        """
        if not self.config.control_encryption or not self.config.control_encryption.encryption_key_length:
            return super().control_encryption_key_length
        return int(self.config.control_encryption.encryption_key_length)

    @control_encryption_key_length.setter
    def control_encryption_key_length(self, value: int) -> None:
        """
        Set the control encryption key length to use.
        """
        if not self.config.control_encryption:
            self.config.control_encryption = EncryptionConfig()
        self.config.control_encryption.encryption_key_length = value

    @property
    def control_encryption_var(self) -> Optional[str]:
        """
        Return environment variable for control encryption key if set, otherwise use default.
        """
        if not self.config.control_encryption or not self.config.control_encryption.encryption_var:
            return super().control_encryption_var
        return str(self.config.control_encryption.encryption_var)

    @control_encryption_var.setter
    def control_encryption_var(self, value: Optional[str]) -> None:
        """
        Set the environment variable for control encryption key to use.
        """
        if not self.config.control_encryption:
            self.config.control_encryption = EncryptionConfig()
        self.config.control_encryption.encryption_var = value

    @property
    def control_encryption_key(self) -> Optional[bytes]:
        """
        Return configured control encryption key if set, otherwise use default.
        """
        if not self.config.control_encryption or not self.config.control_encryption.encryption_key:
            return super().control_encryption_key
        configured = self.config.control_encryption.encryption_key
        if isinstance(configured, bytes):
            return configured
        elif isinstance(configured, str):
            return configured.encode("utf-8")
        return None

    @control_encryption_key.setter
    def control_encryption_key(self, value: Union[bytes, str]) -> None:
        """
        Set the control encryption key to use.
        """
        if not self.config.control_encryption:
            self.config.control_encryption = EncryptionConfig()
        self.config.control_encryption.encryption_key = value.decode("utf-8") if isinstance(value, bytes) else value

    @property
    def allow_list(self) -> Optional[List[str]]:
        """
        Override default allow_list with configured allow_list.
        """
        configured_allow_list = self.config.allow_list
        if configured_allow_list is None:
            return super().allow_list
        return configured_allow_list # type: ignore[no-any-return]

    @allow_list.setter
    def allow_list(self, value: Optional[List[str]]) -> None:
        """
        Set the allow_list to use.
        """
        self.config.allow_list = value

    @property
    def control_list(self) -> Optional[List[str]]:
        """
        Override default control_list with configured control_list.
        """
        configured_control_list = self.config.control_list
        if configured_control_list is None:
            return super().control_list
        return configured_control_list # type: ignore[no-any-return]

    @control_list.setter
    def control_list(self, value: Optional[List[str]]) -> None:
        """
        Set the control_list to use.
        """
        self.config.control_list = value

    @property
    def max_idle_time(self) -> Optional[float]:
        """
        Override default max idle time with configured max idle time.
        """
        if self.config.max_idle_time is None:
            return super().max_idle_time
        return float(self.config.max_idle_time)

    @max_idle_time.setter
    def max_idle_time(self, value: Optional[float]) -> None:
        """
        Set the max idle time to use.
        """
        self.config.max_idle_time = value

    @property
    def certfile(self) -> Optional[str]:
        """
        Override default certfile with configured certfile.
        """
        if not self.config.encryption:
            return super().certfile
        return self.config.encryption.certfile # type: ignore[no-any-return]

    @certfile.setter
    def certfile(self, value: Optional[str]) -> None:
        """
        Set the certfile to use.
        """
        if not self.config.encryption:
            self.config.encryption = EncryptionConfig()
        self.config.encryption.certfile = value
        if hasattr(self, "_ssl_context"):
            delattr(self, "_ssl_context")

    @property
    def keyfile(self) -> Optional[str]:
        """
        Override default keyfile with configured keyfile.
        """
        if not self.config.encryption:
            return super().keyfile
        return self.config.encryption.keyfile # type: ignore[no-any-return]

    @keyfile.setter
    def keyfile(self, value: Optional[str]) -> None:
        """
        Set the keyfile to use.
        """
        if not self.config.encryption:
            self.config.encryption = EncryptionConfig()
        self.config.encryption.keyfile = value
        if hasattr(self, "_ssl_context"):
            delattr(self, "_ssl_context")

    @property
    def cafile(self) -> Optional[str]:
        """
        Override default cafile with configured cafile.
        """
        if not self.config.encryption:
            return super().cafile
        return self.config.encryption.cafile # type: ignore[no-any-return]

    @cafile.setter
    def cafile(self, value: Optional[str]) -> None:
        """
        Set the cafile to use.
        """
        if not self.config.encryption:
            self.config.encryption = EncryptionConfig()
        self.config.encryption.cafile = value
        if hasattr(self, "_ssl_context"):
            delattr(self, "_ssl_context")
