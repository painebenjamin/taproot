from .base import Zephyr7BTextGeneration

__all__ = [
    "Zephyr7BBetaQ80TextGeneration",
    "Zephyr7BBetaQ6KTextGeneration",
    "Zephyr7BBetaQ5KMTextGeneration",
    "Zephyr7BBetaQ4KMTextGeneration"
]

class Zephyr7BBetaQ80TextGeneration(Zephyr7BTextGeneration):
    """
    Zephyr 7B β Text Generation
    """
    task = "text-generation"
    model = "zephyr-7b-beta"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-zephyr-beta-7b-q8-0.gguf"
    display_name = "Zephyr 7B β Text Generation"
    static_memory_gb = 0.95
    static_gpu_memory_gb = 9.4

class Zephyr7BBetaQ6KTextGeneration(Zephyr7BTextGeneration):
    """
    Zephyr 7B β Text Generation (Q6-K)
    """
    task = "text-generation"
    model = "zephyr-7b-beta-q6-k"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-zephyr-beta-7b-q6-k.gguf"
    display_name = "Zephyr 7B β Text Generation (Q6-K)"
    static_memory_gb = 0.95
    static_gpu_memory_gb = 8.2

class Zephyr7BBetaQ5KMTextGeneration(Zephyr7BTextGeneration):
    """
    Zephyr 7B β Text Generation (Q5-K-M)
    """
    task = "text-generation"
    model = "zephyr-7b-beta-q5-k-m"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-zephyr-beta-7b-q5-k-m.gguf"
    display_name = "Zephyr 7B β Text Generation (Q5-K-M)"
    static_memory_gb = 0.95
    static_gpu_memory_gb = 7.25

class Zephyr7BBetaQ4KMTextGeneration(Zephyr7BTextGeneration):
    """
    Zephyr 7B β Text Generation (Q4-K-M)
    """
    task = "text-generation"
    model = "zephyr-7b-beta-q4-k-m"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-zephyr-beta-7b-q4-k-m.gguf"
    display_name = "Zephyr 7B β Text Generation (Q4-K-M)"
    static_memory_gb = 0.95
    static_gpu_memory_gb = 6.3

class Zephyr7BBetaQ3KMTextGeneration(Zephyr7BTextGeneration):
    """
    Zephyr 7B β Text Generation (Q3-K-M)
    """
    task = "text-generation"
    model = "zephyr-7b-beta-q3-k-m"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-zephyr-beta-7b-q3-k-m.gguf"
    display_name = "Zephyr 7B β Text Generation (Q3-K-M)"
    static_memory_gb = 0.95
    static_gpu_memory_gb = 5.35
