from __future__ import annotations

import os
import struct
import pickle

from typing import Optional, Any, Union
from Crypto.Cipher import AES

from .util import (
    logger,
    pack_and_encode,
    decode_and_unpack,
    random_ascii_string
)

__all__ = ["Encryption"]

class Encryption:
    """
    Mixable class for encryption and decryption.

    >>> mixin = Encryption()
    >>> original = "Hello, world!"
    >>> encrypted = mixin.encrypt(original)
    >>> assert(isinstance(encrypted, bytes))
    >>> assert encrypted != original.encode("utf-8")
    >>> decrypted = mixin.decrypt(encrypted)
    >>> assert(decrypted == original)
    """

    """Default properties"""

    @property
    def default_encryption_var(self) -> Optional[str]:
        """
        The default environment variable to use for the encryption key.
        """
        return None

    @property
    def default_encryption_key_length(self) -> int:
        """
        The default length of the encryption key in bytes.
        """
        return 32

    @property
    def default_encryption_use_aesni(self) -> bool:
        """
        Whether to use AES-NI instructions by default.
        """
        return True

    @property
    def default_encryption_key(self) -> bytes:
        """
        The default encryption key used to encrypt and decrypt messages.
        """
        if self.encryption_var:
            try:
                return os.environ[self.encryption_var].encode("utf-8")
            except KeyError:
                logger.warning(
                    f"Environment variable {self.encryption_var} not found. "
                    "Using default encryption key instead."
                )
        logger.info(
            "Using default encryption key, which is random. "
            "This will not be saved or displayed. "
            "If you wish to have server and client as separate processes, "
            "you must set the encryption key manually."
        )
        # Stick to ASCII characters as they can be assured to have one byte
        # per character, so if the key ever needs to be decoded/encoded, the
        # length will be the same.
        return random_ascii_string(self.encryption_key_length).encode("utf-8")

    """Getter/setter properties"""

    @property
    def encryption_var(self) -> Optional[str]:
        """
        The environment variable to use for the encryption key.
        """
        if not hasattr(self, "_encryption_var"):
            self._encryption_var = self.default_encryption_var
        return self._encryption_var

    @encryption_var.setter
    def encryption_var(self, value: Optional[str]) -> None:
        """
        Set the environment variable to use for the encryption key.
        """
        self._encryption_var = value

    @property
    def encryption_key_length(self) -> int:
        """
        The length of the encryption key in bytes.
        """
        if not hasattr(self, "_encryption_key_length"):
            self._encryption_key_length = self.default_encryption_key_length
        return self._encryption_key_length

    @encryption_key_length.setter
    def encryption_key_length(self, value: int) -> None:
        """
        Set the encryption key length.
        """
        self._encryption_key_length = value

    @property
    def encryption_use_aesni(self) -> bool:
        """
        Whether to use AES-NI instructions.
        """
        if not hasattr(self, "_encryption_use_aesni"):
            self._encryption_use_aesni = self.default_encryption_use_aesni
        return self._encryption_use_aesni

    @encryption_use_aesni.setter
    def encryption_use_aesni(self, value: bool) -> None:
        """
        Set whether to use AES-NI instructions.
        """
        self._encryption_use_aesni = value

    @property
    def encryption_key(self) -> bytes:
        """
        The encryption key used to encrypt and decrypt messages.
        """
        if not getattr(self, "_encryption_key", None):
            self._encryption_key = self.default_encryption_key
        return self._encryption_key

    @encryption_key.setter
    def encryption_key(self, value: Union[str, bytes]) -> None:
        """
        Set the encryption key.
        """
        from taproot.util import logger
        if isinstance(value, str):
            value = value.encode("utf-8")
        self._encryption_key = value

    """Internal methods"""

    def _get_encryption_cipher(self, nonce: Optional[bytes] = None) -> object:
        """
        Get the cipher object.
        """
        from taproot.util import logger
        return AES.new(
            self.encryption_key,
            AES.MODE_CTR,
            nonce=nonce,
            use_aesni=self.encryption_use_aesni,
        )

    """Public methods"""

    def encrypt(
        self,
        message: Any,
        use_pack: bool=True,
        use_pickle: bool=False,
    ) -> bytes:
        """
        Encrypt a message.
        Prefixes the message with the length of the nonce and the nonce itself.
        """
        if use_pack:
            if use_pickle:
                logger.warning("Use_pack takes precedence over use_pickle.")
            message = pack_and_encode(message)
        elif use_pickle:
            message = pickle.dumps(message)
        elif isinstance(message, str):
            message = message.encode("utf-8")
        elif not isinstance(message, bytes):
            raise ValueError("Message must be a string, bytes or picklable object.")
        cipher = self._get_encryption_cipher()
        encrypted: bytes = cipher.encrypt(message) # type: ignore[attr-defined]
        nonce: bytes = cipher.nonce # type: ignore[attr-defined]
        nonce_len_packed = struct.pack("<I", len(nonce))
        return nonce_len_packed + nonce + encrypted

    def decrypt(
        self,
        message: bytes,
        use_pack: bool=True,
        use_pickle: bool=False,
    ) -> Any:
        """
        Decrypt a message.
        """
        nonce_len_packed = message[:4]
        nonce_len = struct.unpack("<I", nonce_len_packed)[0]
        nonce = message[4 : 4 + nonce_len]
        cipher = self._get_encryption_cipher(nonce)
        message = cipher.decrypt(message[4 + nonce_len :]) # type: ignore[attr-defined]
        if use_pack:
            if use_pickle:
                logger.warning("Use_pack takes precedence over use_pickle.")
            return decode_and_unpack(message)
        elif use_pickle:
            return pickle.loads(message)
        return message
