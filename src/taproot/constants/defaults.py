import os

__all__ = [
    "DEFAULT_PORT",
    "DEFAULT_HOST",
    "DEFAULT_PROTOCOL",
    "DEFAULT_ADDRESS",
    "DEFAULT_DISPATCHER_PROTOCOL",
    "DEFAULT_DISPATCHER_HOST",
    "DEFAULT_DISPATCHER_PORT",
    "DEFAULT_DISPATCHER_ADDRESS",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_FRAME_RATE",
    "DEFAULT_SAMPLE_RATE",
    "CLIENT_RETRY_DELAY",
    "CLIENT_MAX_RETRIES",
    "SERVER_POLLING_INTERVAL",
    "DISPATCHER_SPAWN_DEBOUNCE_TIME",
    "DISPATCHER_AVAILABILITY_POLLING_INTERVAL",
    "DISPATCHER_RESERVE_MAX_RETRIES",
    "GPU_MEMORY_BANDWIDTH_WEIGHT",
    "GPU_PERFORMANCE_WEIGHT",
    "GPU_MEMORY_UTILIZATION_WEIGHT",
    "GPU_LOAD_WEIGHT",
    "GPU_TOTAL_WEIGHT",
    "CPU_LOAD_WEIGHT",
    "CPU_MEMORY_UTILIZATION_WEIGHT",
    "CPU_TOTAL_WEIGHT",
    "COMBINED_CPU_WEIGHT",
    "COMBINED_GPU_WEIGHT",
    "COMBINED_TOTAL_WEIGHT",
    "AVAILABILITY_SCORE_MAX",
    "STEP_RATE_EMA_ALPHA",
    "WEBSOCKET_CHUNK_SIZE",
    "DEFAULT_MULTIDIFFUSION_MASK_TYPE",
]

DEFAULT_PORT = 32189
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PROTOCOL = "ws"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_RATE = 8
DEFAULT_DISPATCHER_PROTOCOL = "tcp"
DEFAULT_DISPATCHER_HOST = DEFAULT_HOST
DEFAULT_DISPATCHER_PORT = 32190

# Derived, don't edit directly
DEFAULT_ADDRESS = f"{DEFAULT_PROTOCOL}://{DEFAULT_HOST}:{DEFAULT_PORT}"
DEFAULT_DISPATCHER_ADDRESS = f"{DEFAULT_DISPATCHER_PROTOCOL}://{DEFAULT_DISPATCHER_HOST}:{DEFAULT_DISPATCHER_PORT}"

DEFAULT_MODEL_DIR = os.path.expanduser("~/.taproot")
STEP_RATE_EMA_ALPHA = 0.9

CLIENT_MAX_RETRIES = 3
SERVER_POLLING_INTERVAL = 0.1
CLIENT_RETRY_DELAY = 0.05
DISPATCHER_AVAILABILITY_POLLING_INTERVAL = 0.1
DISPATCHER_RESERVE_MAX_RETRIES = 3
DISPATCHER_SPAWN_DEBOUNCE_TIME = 0.2 # Cannot spawn more than one model every 0.2 seconds
WEBSOCKET_CHUNK_SIZE = 1000000  # 1MB

# TODO: experiment with these weights
# GPU ratio weights
GPU_MEMORY_BANDWIDTH_WEIGHT = 0.3
GPU_PERFORMANCE_WEIGHT = 0.5
GPU_MEMORY_UTILIZATION_WEIGHT = 1.0
GPU_LOAD_WEIGHT = 2.0
GPU_TOTAL_WEIGHT = (
    GPU_MEMORY_BANDWIDTH_WEIGHT +
    GPU_PERFORMANCE_WEIGHT +
    GPU_MEMORY_UTILIZATION_WEIGHT +
    GPU_LOAD_WEIGHT
)

# CPU ratio weights
CPU_LOAD_WEIGHT = 2.0
CPU_MEMORY_UTILIZATION_WEIGHT = 1.0
CPU_TOTAL_WEIGHT = CPU_LOAD_WEIGHT + CPU_MEMORY_UTILIZATION_WEIGHT

# Combined CPU/GPU ratio weights
COMBINED_CPU_WEIGHT = 1.0
COMBINED_GPU_WEIGHT = 2.0
COMBINED_TOTAL_WEIGHT = COMBINED_CPU_WEIGHT + COMBINED_GPU_WEIGHT

# Score globals
AVAILABILITY_SCORE_MAX = 10000

try:
    if not os.path.exists(DEFAULT_MODEL_DIR):
        os.makedirs(DEFAULT_MODEL_DIR)
except Exception as e:
    import warnings
    warnings.warn(f"Failed to create model directory {DEFAULT_MODEL_DIR}: {str(e)}")

# Some task globals
DEFAULT_MULTIDIFFUSION_MASK_TYPE = "bilinear"