from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, Union, List, Sequence
from typing_extensions import TypedDict, NotRequired, Literal

__all__ = [
    "ParameterMetadataPayload",
    "CapabilityPayload",
    "TaskMetadataPayload",
    "TaskPayload",
    "ExecutorStatusPayload",
    "ExecutorTargetPayload",
    "ServerStatusPayload",
    "DispatcherStatusPayload",
    "OverseerStatusPayload",
    "FlexibleResultMapping"
]

class ParameterMetadataPayload(TypedDict):
    parameter_type: type
    parameter_size: NotRequired[Optional[Tuple[int,...]]]
    parameter_sub_metadata: NotRequired[
        Optional[
            Union[
                ParameterMetadataPayload,
                Tuple[ParameterMetadataPayload, ...],
                Dict[str, ParameterMetadataPayload]
            ]
        ]
    ]

class CapabilityPayload(TypedDict):
    gpu_memory_bandwidth_gb_s: NotRequired[float]
    gpu_half_float_performance_gflop_s: NotRequired[float]
    gpu_single_float_performance_gflop_s: NotRequired[float]
    gpu_double_float_performance_gflop_s: NotRequired[float]

class TaskMetadataPayload(TypedDict):
    task: str
    id: NotRequired[Optional[str]]
    client_id: NotRequired[Optional[str]]
    model: NotRequired[Optional[str]]
    parameters: NotRequired[Optional[Dict[str, ParameterMetadataPayload]]]
    local_score: NotRequired[int]
    cluster_capability: NotRequired[Optional[CapabilityPayload]]

FlexibleResultMapping = Union[str, Dict[str, 'FlexibleResultMapping'], Sequence['FlexibleResultMapping']]

class TaskPayload(TypedDict):
    task: str
    id: NotRequired[Optional[str]]
    client_id: NotRequired[Optional[str]]
    model: NotRequired[Optional[str]]
    parameters: NotRequired[Optional[Dict[str, Any]]]
    wait_for_result: NotRequired[bool]
    return_metadata: NotRequired[bool]
    continuation: NotRequired[Union[TaskPayload, List[TaskPayload]]]
    overseer: NotRequired[Optional[str]]
    result_parameters: NotRequired[FlexibleResultMapping]

class ExecutorTargetPayload(TypedDict):
    id: str
    address: str
    proxy: NotRequired[Optional[str]]

class ServerStatusPayload(TypedDict):
    active_requests: int
    processing: bool
    uptime: float
    num_requests: int

class ExecutorStatusPayload(ServerStatusPayload):
    allocation: Literal["static", "dynamic"]
    status: Literal["idle", "active", "ready", "zombie", "reserved"]
    activity: float
    capacity: int
    queued: int
    has_id: NotRequired[bool]

class DispatcherStatusPayload(ServerStatusPayload):
    executors: Dict[str, Dict[str, Optional[ExecutorStatusPayload]]]
    overseers: List[str]

class OverseerStatusPayload(ServerStatusPayload):
    dispatchers: Dict[str, Optional[DispatcherStatusPayload]]
