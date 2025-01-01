from __future__ import annotations

import os
import io
import re
import sys
import json
import socket
import random
import msgpack # type: ignore[import-untyped,unused-ignore]
import tempfile
import datetime
import warnings
import subprocess

from contextlib import closing

from typing import Optional, Tuple, Any, Sequence, Dict, Union, TYPE_CHECKING
from typing_extensions import Literal, TypedDict, NotRequired

if TYPE_CHECKING:
    import torch
    import numpy as np
    from PIL.Image import Image
    from ..encryption import Encryption
    from .image_util import EncodedImageProxy
    from .audio_util import EncodedAudioProxy
    from .video_util import EncodedVideoProxy

__all__ = [
    "PackedData",
    "pack_torch_tensor",
    "unpack_torch_tensor",
    "pack_ndarray",
    "unpack_ndarray",
    "pack_image",
    "unpack_image",
    "pack_data",
    "unpack_data",
    "pack_and_encode",
    "decode_and_unpack",
    "is_control_message",
    "pack_control_message",
    "unpack_control_message",
    "parse_address",
    "format_address",
    "generate_temp_key_and_cert",
    "find_free_unix_socket",
    "find_free_memory_port",
    "find_free_port",
    "get_default_ip_address",
]

class PackedData(TypedDict):
    type: Literal["null", "tensor", "ndarray", "image", "int", "float", "bool", "str", "bytes", "list", "dict", "exception", "type", "audio", "video"]
    data: NotRequired[Union[bytes,str,int,float,bool]]
    dtype: NotRequired[str]
    shape: NotRequired[Tuple[int, ...]]
    props: NotRequired[Dict[str, PackedData]]
    items: NotRequired[Sequence[PackedData]]

def pack_exception(exception: Exception) -> PackedData:
    """
    Pack an exception into a packed data dict.
    """
    return {
        "type": "exception",
        "data": str(exception).encode("utf-8"),
        "dtype": type(exception).__name__,
    }

def unpack_exception(packed_data: PackedData) -> Exception:
    """
    Unpack a packed data dict into an exception.
    """
    exception_type = getattr(__builtins__, packed_data["dtype"], Exception)
    exception_data = packed_data["data"]
    if isinstance(exception_data, bytes):
        return exception_type(exception_data.decode("utf-8"))
    return exception_type(exception_data)

def pack_torch_tensor(tensor: torch.Tensor) -> PackedData:
    """
    Pack a torch tensor into a packed data dict.
    """
    as_np = tensor.cpu().numpy()
    return {
        "type": "tensor",
        "dtype": str(as_np.dtype),
        "shape": as_np.shape,
        "data": as_np.tobytes()
    }

def unpack_torch_tensor(packed_data: PackedData) -> torch.Tensor:
    """
    Unpack a packed data dict into a torch tensor.
    """
    assert "data" in packed_data, "tensor type must have data"
    assert isinstance(packed_data["data"], bytes), "tensor data must be bytes"
    assert "dtype" in packed_data, "tensor type must have dtype"
    assert "shape" in packed_data, "tensor type must have shape"
    import torch
    import numpy as np
    return torch.from_numpy(
        np.frombuffer(packed_data["data"], dtype=packed_data["dtype"]).reshape(packed_data["shape"])
    )

def pack_ndarray(ndarray: np.ndarray[Any, Any]) -> PackedData:
    """
    Pack a numpy array into a packed data dict.
    """
    return {
        "type": "ndarray",
        "dtype": str(ndarray.dtype),
        "shape": ndarray.shape,
        "data": ndarray.tobytes()
    }

def unpack_ndarray(packed_data: PackedData) -> np.ndarray[Any, Any]:
    """
    Unpack a packed data dict into a numpy array.
    """
    assert "data" in packed_data, "ndarray type must have data"
    assert isinstance(packed_data["data"], bytes), "ndarray data must be bytes"
    assert "dtype" in packed_data, "ndarray type must have dtype"
    assert "shape" in packed_data, "ndarray type must have shape"
    import numpy as np
    return np.frombuffer(packed_data["data"], dtype=packed_data["dtype"]).reshape(packed_data["shape"])

def pack_image(
    image: Union[Image, EncodedImageProxy]
) -> PackedData:
    """
    Pack a PIL image into a packed data dict.
    """
    from .image_util import EncodedImageProxy
    if isinstance(image, EncodedImageProxy):
        return {
            "type": "image",
            "dtype": image.format, # type: ignore[typeddict-item]
            "shape": image.size,
            "data": image.data
        }

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = buf.getvalue()

    return {
        "type": "image",
        "dtype": "png",
        "shape": image.size,
        "data": buf.getvalue()
    }

def unpack_image(packed_data: PackedData) -> Image:
    """
    Unpack a packed data dict into a PIL image.
    """
    assert "data" in packed_data, "image type must have data"
    assert isinstance(packed_data["data"], bytes), "image data must be bytes"
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 2**32
    return Image.open(io.BytesIO(packed_data["data"]))

def pack_data(mixed_data: Any, encode_strings: bool=True) -> PackedData:
    """
    Pack mixed data into a bytes object.
    """
    from .introspection_util import is_torch_tensor, is_numpy_array, is_pil_image
    from .image_util import EncodedImageProxy
    from .audio_util import EncodedAudioProxy
    from .video_util import EncodedVideoProxy
    if mixed_data is None:
        return {"type": "null"}
    elif isinstance(mixed_data, type):
        if is_pil_image(mixed_data):
            return {"type": "type", "dtype": "Image"}
        elif is_torch_tensor(mixed_data):
            return {"type": "type", "dtype": "Tensor"}
        elif is_numpy_array(mixed_data):
            return {"type": "type", "dtype": "NDArray"}
        else:
            return {"type": "type", "dtype": mixed_data.__name__}
    elif type(mixed_data) in (list, tuple):
        return {"type": "list", "items": [pack_data(item, encode_strings=encode_strings) for item in mixed_data]}
    elif type(mixed_data) is dict:
        return {"type": "dict", "props": {key: pack_data(value, encode_strings=encode_strings) for key, value in mixed_data.items()}}
    elif is_torch_tensor(mixed_data):
        return pack_torch_tensor(mixed_data)
    elif is_numpy_array(mixed_data):
        return pack_ndarray(mixed_data)
    elif is_pil_image(mixed_data) or isinstance(mixed_data, EncodedImageProxy):
        return pack_image(mixed_data)
    elif isinstance(mixed_data, EncodedAudioProxy):
        return {"type": "audio", "data": mixed_data.data, "dtype": mixed_data.format}
    elif isinstance(mixed_data, EncodedVideoProxy):
        return {"type": "video", "data": mixed_data.data, "dtype": mixed_data.format}
    elif isinstance(mixed_data, Exception):
        return pack_exception(mixed_data)
    elif type(mixed_data) in (int, float, bool):
        return {"type": type(mixed_data).__name__, "data": mixed_data} # type: ignore[typeddict-item]
    elif type(mixed_data) == str:
        return {"type": "str", "data": mixed_data.encode("utf-8") if encode_strings else mixed_data}
    elif type(mixed_data) in [bytes, bytearray]: # type: ignore[list-item]
        return {"type": "bytes", "data": mixed_data}
    else:
        raise ValueError(f"Unsupported data type: {type(mixed_data).__name__}")

def unpack_data(packed_data: PackedData, decode_strings: bool=True) -> Any:
    """
    Unpack a bytes object into mixed data.
    """
    from .image_util import EncodedImageProxy
    from .audio_util import EncodedAudioProxy
    if packed_data["type"] == "null":
        return None
    elif packed_data["type"] == "type":
        if packed_data["dtype"] in ["ndarray", "NDArray"]:
            import numpy as np
            return np.ndarray
        elif packed_data["dtype"] in ["tensor", "Tensor"]:
            import torch
            return torch.Tensor
        elif packed_data["dtype"] in ["image", "Image"]:
            from PIL import Image
            return Image.Image
        elif packed_data["dtype"] == "NoneType":
            return type(None)
        elif packed_data["dtype"] == "EncodedAudioProxy":
            return EncodedAudioProxy
        elif packed_data["dtype"] == "EncodedImageProxy":
            return EncodedImageProxy
        return getattr(sys.modules["builtins"], packed_data["dtype"])
    elif packed_data["type"] == "list":
        assert "items" in packed_data, "list type must have items"
        return [unpack_data(item, decode_strings=decode_strings) for item in packed_data["items"]]
    elif packed_data["type"] == "dict":
        assert "props" in packed_data, "dict type must have props"
        return {key: unpack_data(value, decode_strings=decode_strings) for key, value in packed_data["props"].items()}
    elif packed_data["type"] == "tensor":
        return unpack_torch_tensor(packed_data)
    elif packed_data["type"] == "ndarray":
        return unpack_ndarray(packed_data)
    elif packed_data["type"] == "image":
        return unpack_image(packed_data)
    elif packed_data["type"] == "audio":
        from .audio_util import EncodedAudioProxy
        return EncodedAudioProxy(packed_data["data"], packed_data["dtype"]) # type: ignore[arg-type]
    elif packed_data["type"] == "video":
        from .video_util import EncodedVideoProxy
        return EncodedVideoProxy(packed_data["data"], packed_data["dtype"]) # type: ignore[arg-type]
    elif packed_data["type"] == "exception":
        return unpack_exception(packed_data)
    elif packed_data["type"] == "int":
        return int(packed_data["data"])
    elif packed_data["type"] == "float":
        return float(packed_data["data"])
    elif packed_data["type"] == "bool":
        return bool(packed_data["data"])
    elif packed_data["type"] == "str":
        if decode_strings and isinstance(packed_data["data"], bytes):
            return packed_data["data"].decode("utf-8")
        return packed_data["data"]
    elif packed_data["type"] == "bytes":
        return packed_data["data"]
    else:
        raise ValueError(f"Unsupported data type: {packed_data['type']}")

def pack_and_encode(data: Any) -> bytes:
    """
    Pack and encode mixed data into a bytes object.
    """
    return msgpack.packb(pack_data(data)) # type: ignore[no-any-return]

def decode_and_unpack(data: bytes) -> Any:
    """
    Decode and unpack a bytes object into mixed data.
    """
    return unpack_data(msgpack.unpackb(data))

def is_control_message(message: Any) -> bool:
    """
    Determine if a payload is a control payload.
    """
    if not isinstance(message, str):
        return False
    return message.startswith("control:")

def pack_control_message(
    message: str,
    data: Any=None,
    encryption: Optional[Encryption]=None
) -> str:
    """
    Packs a control message and metadata into a string.
    We use JSON for ease-of-use and readability of logs,
    these will always only contain small bits of metadata.
    """
    now_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp()*1000)
    message = f"{now_time}:{message}"
    if encryption:
        message = encryption.encrypt(message).hex()

    control_message = f"control:{message}"

    if data is None:
        return control_message

    if type(data) is str:
        data_str = data
    else:
        data_str = json.dumps(pack_data(data, encode_strings=False))

    if encryption:
        data_str = encryption.encrypt(data_str).hex()

    control_message = f"{control_message}:{data_str}"
    return control_message

def unpack_control_message(
    message: str,
    assert_age: Optional[int]=300000, # 5 minutes
    encryption: Optional[Encryption]=None
) -> Tuple[str, Any]:
    """
    Unpacks a control message and metadata from a string.
    """
    message_parts = message.split(":")
    if message_parts[0] != "control":
        raise ValueError(f"Invalid control message - expected 'control', got {message_parts[0]}")

    data: Optional[str] = None
    if encryption:
        message = message_parts[1]
        if len(message_parts) > 2:
            data = ":".join(message_parts[2:])

        if encryption:
            message = encryption.decrypt(bytes.fromhex(message))
            if data:
                data = encryption.decrypt(bytes.fromhex(data))

        timestamp_str, message = message.split(":", 1)
    else:
        timestamp_str = message_parts[1]
        message = message_parts[2]
        if len(message_parts) > 3:
           data = ":".join(message_parts[3:])

    timestamp = int(timestamp_str)
    now_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp()*1000)
    if assert_age and now_time - timestamp > assert_age:
        raise ValueError(f"Control message expired: {now_time - timestamp}ms > {assert_age}ms")

    if not data:
        return message, None
    elif data.startswith("{") and data.endswith("}"):
        return message, unpack_data(json.loads(data), decode_strings=False)
    return message, data

def generate_temp_key_and_cert(
    country: str="US",
    state: str="CA",
    location: str="Tempfile Town",
    organization: str="OpenSSL LLC",
    domain: str="localhost",
    ip_address: Optional[str]="127.0.0.1",
) -> Tuple[str, str]:
    """
    Generate a temporary key and certificate for testing purposes using the command:
    $ openssl req -new -x509 -days 365 -nodes -out cert.pem -keyout key.pem
    """
    subj = f"/C={country}/ST={state}/L={location}/O={organization}/CN={domain}"
    key_path = tempfile.mktemp(prefix="key_", suffix=".pem")
    cert_path = tempfile.mktemp(prefix="cert_", suffix=".pem")
    args = ["openssl", "req", "-new", "-x509", "-days", "365", "-nodes", "-out", cert_path, "-keyout", key_path, "-subj", subj]
    if ip_address:
        args += ["-addext", f"subjectAltName=IP:{ip_address}"]
    subprocess.run(args, check=True)
    return key_path, cert_path

def find_free_unix_socket() -> str:
    """
    Find a free UNIX socket path on the local machine.
    """
    return tempfile.mktemp(prefix="socket_")

def find_free_port() -> int:
    """
    Find a free port on the local machine.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])

ADDRESS_REGEX = re.compile(r"^(\w+)\:\/\/([\w\-._~!#$%^&*\(\)\ ;=]*)(\:(\d+))?(\/.*)?$")

class AddressDict(TypedDict):
    scheme: Literal["memory", "unix", "tcp", "tcps", "ws", "wss"]
    host: Optional[str]
    port: Optional[int]
    path: Optional[str]

def parse_address(address: str) -> AddressDict:
    """
    Parse an address string into a dict of scheme, host, and port.

    >>> parse_address("memory://1234")
    {'scheme': 'memory', 'host': None, 'port': 1234, 'path': None}
    >>> parse_address("tcp://localhost:1234")
    {'scheme': 'tcp', 'host': 'localhost', 'port': 1234, 'path': None}
    >>> parse_address("unix:///tmp/foo")
    {'scheme': 'unix', 'host': None, 'port': None, 'path': '/tmp/foo'}
    """
    match = ADDRESS_REGEX.match(address.strip(" /\r\n\t"))
    if not match:
        raise ValueError(f"Invalid address: {address}")

    scheme: str = match.group(1)
    host: Optional[str] = match.group(2)
    port: Optional[int] = None
    port_str: Optional[str] = None
    path: Optional[str] = None

    try:
        port_str = match.group(4)
    except IndexError:
        pass

    try:
        path = match.group(5)
    except IndexError:
        pass

    if scheme not in ["memory", "unix", "tcp", "tcps", "ws", "wss"]:
        raise ValueError(f"Invalid scheme: {scheme}")

    if scheme == "memory":
        if host and port_str:
            warnings.warn("Ignoring host for memory scheme")
            port = int(port_str)
        elif host:
            try:
                port = int(host)
            except ValueError:
                raise ValueError("Memory scheme requires a numeric port")
        else:
            port = None
        host = None
    elif scheme == "unix":
        if host:
            # User provided a relative path
            path_base = os.path.join(os.getcwd(), host)
            if path:
                path = os.path.join(path_base, path)
            else:
                path = path_base
        if port_str:
            warnings.warn("Ignoring port for unix scheme")
        host = None
        port = None
    elif scheme in ["tcp", "tcps", "ws", "wss"]:
        if port_str:
            port = int(port_str)
        else:
            port = None

    return {"scheme": scheme, "host": host, "port": port, "path": path} # type: ignore[typeddict-item]

def format_address(address: AddressDict) -> str:
    """
    Formats an address dict into a string.
    """
    base = f"{address['scheme']}://"
    if address["scheme"] == "memory":
        return f"{base}{address['port']}"
    elif address["scheme"] == "unix":
        return f"{base}{address['path']}"
    path = address["path"] or ""
    host_port = f"{address['host']}:{address['port']}" if address["port"] else address["host"]
    return f"{base}{host_port}{path}"

ALL_MEMORY_PORTS = set(range(65536))
def find_free_memory_port() -> int:
    """
    Get a random free port number for in-memory server.
    """
    from ..server.memory import IN_MEMORY_SERVERS
    available_ports = ALL_MEMORY_PORTS - set(IN_MEMORY_SERVERS.keys())
    return random.choice(list(available_ports))

def get_default_ip_address(
    method: Literal["dns", "route"]="route"
) -> str:
    """
    Get the default IP address of the local machine.
    """
    if method == "route":
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(("8.8.8.8", 1))
            ip_address = s.getsockname()[0]
        except:
            ip_address = "127.0.0.1"
        finally:
            s.close()
        return str(ip_address)
    elif method == "dns":
        return socket.gethostbyname(socket.gethostname())
    raise ValueError(f"Invalid method: {method}")
