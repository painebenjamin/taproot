from .base import Llama3TextGeneration

__all__ = [
    "LlamaV30TextGeneration8BQ80",
    "LlamaV30TextGeneration8BQ6K",
    "LlamaV30TextGeneration8BQ5KM",
    "LlamaV30TextGeneration8BQ4KM",
    "LlamaV30TextGeneration8BInstructQ80",
    "LlamaV30TextGeneration8BInstructQ6K",
    "LlamaV30TextGeneration8BInstructQ5KM",
    "LlamaV30TextGeneration8BInstructQ4KM",
]

class LlamaV30TextGeneration8BQ80(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B model with Q8-0 quantization.
    """
    task = "text-generation"
    model = "llama-v3-8b"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-8b-q8-0.gguf"
    display_name = "Llama V3.0 8B Text Generation"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 9.64

class LlamaV30TextGeneration8BQ6K(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B model with Q6-K quantization.
    """
    task = "text-generation"
    model = "llama-v3-8b-q6-k"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-8b-q6-k.gguf"
    display_name = "Llama V3.0 8B Text Generation (Q6-K)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 8.1

class LlamaV30TextGeneration8BQ5KM(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B model with Q5-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-8b-q5-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-8b-q5-k-m.gguf"
    display_name = "Llama V3.0 8B Text Generation (Q5-K-M)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 7.3

class LlamaV30TextGeneration8BQ4KM(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B model with Q4-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-8b-q4-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-8b-q4-k-m.gguf"
    display_name = "Llama V3.0 8B Text Generation (Q4-K-M)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 6.56

class LlamaV30TextGeneration8BQ3KM(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B model with Q3-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-8b-q3-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-8b-q3-k-m.gguf"
    display_name = "Llama V3.0 8B Text Generation (Q3-K-M)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 5.72

class LlamaV30TextGeneration8BInstructQ80(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B instruct model with Q8-0 quantization.
    """
    task = "text-generation"
    model = "llama-v3-8b-instruct"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-8b-instruct-q8-0.gguf"
    display_name = "Llama V3.0 8B Instruct Text Generation"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 9.64

class LlamaV30TextGeneration8BInstructQ6K(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B instruct model with Q6-K quantization.
    """
    task = "text-generation"
    model = "llama-v3-8b-instruct-q6-k"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-8b-instruct-q6-k.gguf"
    display_name = "Llama V3.0 8B Instruct Text Generation (Q6-K)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 8.1

class LlamaV30TextGeneration8BInstructQ5KM(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B instruct model with Q5-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-8b-instruct-q5-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-8b-instruct-q5-k-m.gguf"
    display_name = "Llama V3.0 8B Instruct Text Generation (Q5-K-M)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 7.3

class LlamaV30TextGeneration8BInstructQ4KM(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B instruct model with Q4-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-8b-instruct-q4-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-8b-instruct-q4-k-m.gguf"
    display_name = "Llama V3.0 8B Instruct Text Generation (Q4-K-M)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 6.56

class LlamaV30TextGeneration8BInstructQ3KM(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 8B instruct model with Q3-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-8b-instruct-q3-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-8b-instruct-q3-k-m.gguf"
    display_name = "Llama V3.0 8B Instruct Text Generation (Q3-K-M)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 5.72
