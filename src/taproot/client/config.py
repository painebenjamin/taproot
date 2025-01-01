from __future__ import annotations

from typing import Literal, Union, Optional

from ..config import *
from .base import Client

__all__ = [
    "ConfigClient",
    "DispatcherConfigClient",
]

class ConfigClient(Client, ConfigMixin):
    """
    Client class that uses a configuration file to set default values.
    """
    config_class = ClientConfig

    @property
    def default_use_encryption(self) -> bool:
        """
        Override default encryption setting with configured encryption setting.
        """
        return self.config.encryption is not None

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
    def certfile(self) -> Optional[str]:
        """
        Return configured certfile if set, otherwise use default.
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
    def cafile(self) -> Optional[str]:
        """
        Return configured cafile if set, otherwise use default.
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

    @property
    def default_ca(self) -> bool:
        """
        Whether to load default verification locations.
        """
        if not self.config.encryption:
            return super().default_ca
        return bool(self.config.encryption.default_ca)

    @default_ca.setter
    def default_ca(self, value: bool) -> None:
        """
        Set whether to load default verification locations.
        """
        if not self.config.encryption:
            self.config.encryption = EncryptionConfig()
        self.config.encryption.default_ca = value
        if hasattr(self, "_ssl_context"):
            delattr(self, "_ssl_context")

    @property
    def certifi_ca(self) -> bool:
        """
        Whether to load the certifi CA.
        """
        if not self.config.encryption:
            return super().certifi_ca
        return bool(self.config.encryption.certifi_ca)

    @certifi_ca.setter
    def certifi_ca(self, value: bool) -> None:
        """
        Set whether to load the certifi CA.
        """
        if not self.config.encryption:
            self.config.encryption = EncryptionConfig()
        self.config.encryption.certifi_ca = value
        if hasattr(self, "_ssl_context"):
            delattr(self, "_ssl_context")

class DispatcherConfigClient(ConfigClient):
    """
    Client class that uses a configuration file to set default values.
    """
    config_class = DispatcherClientConfig
