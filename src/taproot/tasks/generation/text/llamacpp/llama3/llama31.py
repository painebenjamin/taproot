from .base import Llama3TextGeneration

__all__ = [
    "LlamaV31TextGeneration8BInstructQ80",
    "LlamaV31TextGeneration8BInstructQ6K",
    "LlamaV31TextGeneration8BInstructQ5KM",
    "LlamaV31TextGeneration8BInstructQ4KM",
    "LlamaV31TextGeneration8BInstructQ3KM",
]

class LlamaV31TextGeneration8BInstructQ80(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B instruct model with Q8-0 quantization.
    """
    task = "text-generation"
    model = "llama-v3-1-8b-instruct"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-1-8b-instruct-q8-0.gguf"
    display_name = "Llama V3.1 8B Instruct Text Generation"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 9.64
    max_context_length = 131072

class LlamaV31TextGeneration8BInstructQ6K(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B instruct model with Q6-K quantization.
    """
    task = "text-generation"
    model = "llama-v3-1-8b-instruct-q6-k"
    default = True
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-1-8b-instruct-q6-k.gguf"
    display_name = "Llama V3.1 8B Instruct Text Generation (Q6-K)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 8.1
    max_context_length = 131072

class LlamaV31TextGeneration8BInstructQ5KM(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B instruct model with Q5-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-1-8b-instruct-q5-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-1-8b-instruct-q5-k-m.gguf"
    display_name = "Llama V3.1 8B Instruct Text Generation (Q5-K-M)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 7.3
    max_context_length = 131072

class LlamaV31TextGeneration8BInstructQ4KM(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B instruct model with Q4-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-1-8b-instruct-q4-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-1-8b-instruct-q4-k-m.gguf"
    display_name = "Llama V3.1 8B Instruct Text Generation (Q4-K-M)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 6.56
    max_context_length = 131072

class LlamaV31TextGeneration8BInstructQ3KM(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B instruct model with Q3-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-1-8b-instruct-q3-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-1-8b-instruct-q3-k-m.gguf"
    display_name = "Llama V3.1 8B Instruct Text Generation (Q3-K-M)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 5.72
    max_context_length = 131072
