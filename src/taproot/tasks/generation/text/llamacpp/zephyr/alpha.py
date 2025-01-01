from .base import Zephyr7BTextGeneration

__all__ = [
    "Zephyr7BAlphaQ80TextGeneration",
    "Zephyr7BAlphaQ6KTextGeneration",
    "Zephyr7BAlphaQ5KMTextGeneration",
    "Zephyr7BAlphaQ4KMTextGeneration",
    "Zephyr7BAlphaQ3KMTextGeneration"
]

class Zephyr7BAlphaQ80TextGeneration(Zephyr7BTextGeneration):
    """
    Zephyr 7B α Text Generation (Q8)
    """
    task = "text-generation"
    model = "zephyr-7b-alpha"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-zephyr-alpha-7b-q8-0.gguf"
    display_name = "Zephyr 7B α Text Generation (Q8)"
    static_memory_gb = 1.1
    static_gpu_memory_gb = 9.4

class Zephyr7BAlphaQ6KTextGeneration(Zephyr7BTextGeneration):
    """
    Zephyr 7B α Text Generation (Q6-K)
    """
    task = "text-generation"
    model = "zephyr-7b-alpha-q6-k"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-zephyr-alpha-7b-q6-k.gguf"
    display_name = "Zephyr 7B α Text Generation (Q6-K)"
    static_memory_gb = 1.1
    static_gpu_memory_gb = 8.2

class Zephyr7BAlphaQ5KMTextGeneration(Zephyr7BTextGeneration):
    """
    Zephyr 7B α Text Generation (Q5-K-M)
    """
    task = "text-generation"
    model = "zephyr-7b-alpha-q5-k-m"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-zephyr-alpha-7b-q5-k-m.gguf"
    display_name = "Zephyr 7B α Text Generation (Q5-K-M)"
    static_memory_gb = 1.1
    static_gpu_memory_gb = 7.25

class Zephyr7BAlphaQ4KMTextGeneration(Zephyr7BTextGeneration):
    """
    Zephyr 7B α Text Generation (Q4-K-M)
    """
    task = "text-generation"
    model = "zephyr-7b-alpha-q4-k-m"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-zephyr-alpha-7b-q4-k-m.gguf"
    display_name = "Zephyr 7B α Text Generation (Q4-K-M)"
    static_memory_gb = 1.1
    static_gpu_memory_gb = 6.3

class Zephyr7BAlphaQ3KMTextGeneration(Zephyr7BTextGeneration):
    """
    Zephyr 7B α Text Generation (Q3-K-M)
    """
    task = "text-generation"
    model = "zephyr-7b-alpha-q3-k-m"
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-zephyr-alpha-7b-q3-k-m.gguf"
    display_name = "Zephyr 7B α Text Generation (Q3-K-M)"
    static_memory_gb = 1.1
    static_gpu_memory_gb = 5.35
