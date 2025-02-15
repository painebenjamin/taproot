import os

__all__ = [
    "HUGGINGFACE_DOMAINS",
    "DEFAULT_PREBUILT_CUDA_VERSION",
    "PREBUILT_CUDA_VERSION",
    "NVIDIA_REPO_URL",
    "TORCH_REPO_URL",
    "TORCH_REPO_URL_BASE",
    "LLAMA_CPP_REPO_URL",
    "LLAMA_CPP_REPO_URL_BASE",
]

HUGGINGFACE_DOMAINS = [
    "huggingface.com",
    "huggingface.co",
    "hf.co",
    "hf.space",
]

DEFAULT_PREBUILT_CUDA_VERSION = "cu124"
NVIDIA_REPO_URL = "https://pypi.ngc.nvidia.com"
TORCH_REPO_URL_BASE = "https://download.pytorch.org/whl"
LLAMA_CPP_REPO_URL_BASE = "https://abetlen.github.io/llama-cpp-python/whl"
PREBUILT_CUDA_VERSION = os.getenv("PREBUILT_CUDA_VERSION", DEFAULT_PREBUILT_CUDA_VERSION)
TORCH_REPO_URL = f"{TORCH_REPO_URL_BASE}/{PREBUILT_CUDA_VERSION}"
LLAMA_CPP_REPO_URL = f"{LLAMA_CPP_REPO_URL_BASE}/{PREBUILT_CUDA_VERSION}"
