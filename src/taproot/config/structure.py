from __future__ import annotations

from omegaconf import MISSING
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from typing_extensions import TypedDict, NotRequired

from ..constants import *

__all__ = [
    "EncryptionConfig",
    "EncryptionConfigDict",
    "ServerConfig",
    "ServerConfigDict",
    "TaskConfig",
    "TaskConfigDict",
    "TaskQueueConfig",
    "TaskQueueConfigDict",
    "ExecutorConfig",
    "ExecutorConfigDict",
    "DispatcherConfig",
    "DispatcherConfigDict",
    "OverseerConfig",
    "OverseerConfigDict",
    "ClientConfig",
    "ClientConfigDict",
    "DispatcherClientConfig",
    "DispatcherClientConfigDict",
    "TapConfig",
    "TapConfigDict"
]

@dataclass
class EncryptionConfig:
    """
    Configuration for encryption on servers and clients.
    """
    encryption_key: Optional[Union[str, bytes]] = None # Encryption key
    encryption_var: Optional[str] = None # Environment variable to read encryption key from
    encryption_key_length: int = 32 # 256 bits, only used if encryption_key is None
    encryption_use_aesni: bool = True # Use AES-NI instructions if available
    certfile: Optional[str] = None # Certificate file
    keyfile: Optional[str] = None # Key file
    cafile: Optional[str] = None # CA file
    default_ca: bool = True # Use default CA certificates (system dependent)
    certifi_ca: bool = True # Use certifi to locate CA file

class EncryptionConfigDict(TypedDict):
    """
    TypedDict for EncryptionConfig.
    """
    encryption_key: NotRequired[Optional[Union[str, bytes]]]
    encryption_var: NotRequired[Optional[str]]
    encryption_key_length: NotRequired[int]
    encryption_use_aesni: NotRequired[bool]
    certfile: NotRequired[Optional[str]]
    keyfile: NotRequired[Optional[str]]
    cafile: NotRequired[Optional[str]]
    default_ca: NotRequired[bool]
    certifi_ca: NotRequired[bool]

@dataclass
class ServerConfig:
    """
    Configuration for servers.
    """
    protocol: str = "ws"
    host: Optional[str] = None
    port: Optional[int] = None
    path: Optional[str] = None
    external_address: Optional[str] = None
    encryption: Optional[EncryptionConfig] = None
    control_encryption: Optional[EncryptionConfig] = None
    allow_list: Optional[List[str]] = None
    control_list: Optional[List[str]] = None
    max_idle_time: Optional[float] = None

class ServerConfigDict(TypedDict):
    """
    TypedDict for ServerConfig.
    """
    protocol: PROTOCOL_LITERAL
    host: NotRequired[Optional[str]]
    port: NotRequired[Optional[int]]
    path: NotRequired[Optional[str]]
    external_address: NotRequired[Optional[str]]
    encryption: NotRequired[Optional[EncryptionConfigDict]]
    control_encryption: NotRequired[Optional[EncryptionConfigDict]]
    allow_list: NotRequired[Optional[List[str]]]
    control_list: NotRequired[Optional[List[str]]]
    max_idle_time: NotRequired[Optional[float]]

@dataclass
class TaskConfig:
    """
    Configuration for tasks.
    """
    gpu_index: Optional[int] = None
    dtype: Optional[str] = None
    model_dir: Optional[str] = None
    rate_ema_alpha: Optional[float] = None
    save_dir: Optional[str] = None
    save_format: Optional[str] = None
    compile_pretrained: bool = False
    allow_optional: bool = True
    use_tqdm: bool = False
    context_length: Optional[int] = None
    enable_model_offload: Optional[bool] = None
    enable_sequential_offload: Optional[bool] = None
    enable_encode_tiling: Optional[bool] = None
    enable_encode_slicing: Optional[bool] = None
    options: Dict[str, Any] = field(default_factory = dict)

class TaskConfigDict(TypedDict):
    """
    TypedDict for TaskConfig.
    """
    gpu_index: NotRequired[Optional[int]]
    dtype: NotRequired[Optional[str]]
    model_dir: NotRequired[Optional[str]]
    rate_ema_alpha: NotRequired[float]
    save_dir: NotRequired[Optional[str]]
    compile_pretrained: NotRequired[bool]
    allow_optional: NotRequired[bool]
    use_tqdm: NotRequired[bool]
    context_length: NotRequired[Optional[int]]
    enable_model_offload: NotRequired[Optional[bool]]
    enable_sequential_offload: NotRequired[Optional[bool]]
    enable_encode_tiling: NotRequired[Optional[bool]]
    enable_encode_slicing: NotRequired[Optional[bool]]
    options: NotRequired[Dict[str, Any]]

@dataclass
class TaskQueueConfig:
    """
    Configuration for task queues.
    """
    task: str = MISSING
    model: Optional[str] = None
    task_config: TaskConfig = field(default_factory = TaskConfig)
    polling_interval: float = 0.01
    result_duration: Optional[float] = None
    size: int = 1
    activity_tau: float = 30.0 # Seconds

class TaskQueueConfigDict(TypedDict):
    """
    TypedDict for TaskQueueConfig.
    """
    task: str
    model: NotRequired[Optional[str]]
    task_config: NotRequired[TaskConfigDict]
    polling_interval: NotRequired[float]
    result_duration: NotRequired[Optional[float]]
    size: NotRequired[int]
    activity_tau: NotRequired[float]

@dataclass
class ExecutorConfig(ServerConfig):
    """
    Configuration for the executor (phase 3) server.
    """
    queue_config: TaskQueueConfig = field(default_factory = TaskQueueConfig)
    install: bool = False
    allocation: str = "dynamic"
    max_reservation_time: float = 0.5
    max_continuation_depth: int = 10

class ExecutorConfigDict(ServerConfigDict):
    """
    TypedDict for ExecutorConfig.
    """
    queue_config: NotRequired[TaskQueueConfigDict]
    install: NotRequired[bool]
    allocation: NotRequired[str]
    max_reservation_time: NotRequired[float]
    max_continuation_depth: NotRequired[int]

@dataclass
class DispatcherConfig(ServerConfig):
    """
    Configuration for the dispatcher (phase 2) server.
    """
    executor_config: ExecutorConfig = field(default_factory = ExecutorConfig) # Configuration for each executor
    static_executor_config: List[ExecutorConfig] = field(default_factory = list) # Statically allocated executors
    max_workers: Optional[int] = None # Maximum total number of workers
    use_multiprocessing: bool = False # Use multiprocessing for workers, default is threading
    spawn_interval: Optional[float] = 5.0 # Interval to spawn workers
    task_max_workers: Optional[Dict[str, int]] = None # Maximum number of workers for each task
    task_config: Optional[Dict[str, TaskQueueConfig]] = None # Per-task configuration
    task_allow_list: Optional[List[str]] = None # Allowed tasks
    task_denylist: Optional[List[str]] = None # Disallowed tasks

class DispatcherConfigDict(ServerConfigDict):
    """
    TypedDict for DispatcherConfig.
    """
    executor_config: NotRequired[ExecutorConfigDict]
    static_executor_config: NotRequired[List[ExecutorConfigDict]]
    max_workers: NotRequired[Optional[int]]
    use_multiprocessing: NotRequired[bool]
    spawn_interval: NotRequired[Optional[float]]
    task_max_workers: NotRequired[Optional[Dict[str, int]]]
    task_config: NotRequired[Optional[Dict[str, TaskQueueConfigDict]]]
    task_allow_list: NotRequired[Optional[List[str]]]
    task_denylist: NotRequired[Optional[List[str]]]

@dataclass
class OverseerConfig(ServerConfig):
    """
    Configuration for the overseer (phase 1) server.
    """
    dispatchers: List[DispatcherClientConfig] = field(default_factory = list)
    dispatcher_score_timeout: Optional[float] = None
    dispatcher_prepare_timeout: Optional[float] = None

class OverseerConfigDict(ServerConfigDict):
    """
    TypedDict for OverseerConfig.
    """
    dispatchers: NotRequired[List[DispatcherClientConfigDict]]
    dispatcher_score_timeout: NotRequired[Optional[float]]
    dispatcher_prepare_timeout: NotRequired[Optional[float]]

@dataclass
class ClientConfig:
    """
    Configuration for clients.
    """
    protocol: str = "memory"
    host: Optional[str] = None
    port: Optional[int] = None
    path: Optional[str] = None
    encryption: Optional[EncryptionConfig] = None
    certfile: Optional[str] = None
    cafile: Optional[str] = None

class ClientConfigDict(TypedDict):
    """
    TypedDict for ClientConfig.
    """
    protocol: PROTOCOL_LITERAL
    host: NotRequired[Optional[str]]
    port: NotRequired[Optional[int]]
    path: NotRequired[Optional[str]]
    encryption: NotRequired[Optional[EncryptionConfigDict]]
    certfile: NotRequired[Optional[str]]
    cafile: NotRequired[Optional[str]]

@dataclass
class DispatcherClientConfig(ClientConfig):
    """
    Configuration for the dispatcher client.
    """
    resolve_addresses: bool = True

class DispatcherClientConfigDict(ClientConfigDict):
    """
    TypedDict for DispatcherClientConfig.
    """
    resolve_addresses: NotRequired[bool]

@dataclass
class TapConfig:
    """
    Configuration for the Tap (flexible client)
    """
    remote: Optional[ClientConfig] = None
    local: Optional[DispatcherConfig] = None

class TapConfigDict(TypedDict):
    """
    TypedDict for TapConfig.
    """
    remote: NotRequired[Optional[ClientConfigDict]]
    local: NotRequired[Optional[DispatcherConfigDict]]
