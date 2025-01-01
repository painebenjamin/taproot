from .base import Llama3TextGeneration

__all__ = [
    "LlamaV32TextGeneration3BInstruct",
    "LlamaV32TextGeneration3BInstructQ80",
    "LlamaV32TextGeneration3BInstructQ6K",
    "LlamaV32TextGeneration3BInstructQ5KM",
    "LlamaV32TextGeneration3BInstructQ4KM",
    "LlamaV32TextGeneration3BInstructQ3KL",
]

class LlamaV32TextGeneration3BInstruct(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 3B instruct model with no quantization.
    """
    task = "text-generation"
    model = "llama-v3-2-3b-instruct"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-2-3b-instruct-f16.gguf"
    display_name = "Llama V3.2 3B Instruct Text Generation"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 8.04
    max_context_length = 131072

class LlamaV32TextGeneration3BInstructQ80(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 3B instruct model with Q8-0 quantization.
    """
    task = "text-generation"
    model = "llama-v3-2-3b-instruct-q8-0"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-2-3b-instruct-q8-0.gguf"
    display_name = "Llama V3.2 3B Instruct Text Generation (Q8-0)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 5.02
    max_context_length = 131072

class LlamaV32TextGeneration3BInstructQ6K(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 3B instruct model with Q6-K quantization.
    """
    task = "text-generation"
    model = "llama-v3-2-3b-instruct-q6-k"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-2-3b-instruct-q6-k.gguf"
    display_name = "Llama V3.2 3B Instruct Text Generation (Q6-K)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 4.2
    max_context_length = 131072

class LlamaV32TextGeneration3BInstructQ5KM(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 3B instruct model with Q5-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-2-3b-instruct-q5-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-2-3b-instruct-q5-k-m.gguf"
    display_name = "Llama V3.2 3B Instruct Text Generation (Q5-K-M)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 3.9
    max_context_length = 131072

class LlamaV32TextGeneration3BInstructQ4KM(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 3B instruct model with Q4-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-2-3b-instruct-q4-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-2-3b-instruct-q4-k-m.gguf"
    display_name = "Llama V3.2 3B Instruct Text Generation (Q4-K-M)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 3.5
    max_context_length = 131072

class LlamaV32TextGeneration3BInstructQ3KL(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 3B instruct model with Q3-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-2-3b-instruct-q3-k-l"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-2-3b-instruct-q3-k-l.gguf"
    display_name = "Llama V3.2 3B Instruct Text Generation (Q3-K-L)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 3.1
    max_context_length = 131072

class LlamaV32TextGeneration1BInstruct(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 1B instruct model with no quantization.
    """
    task = "text-generation"
    model = "llama-v3-2-1b-instruct"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-2-1b-instruct-f16.gguf"
    display_name = "Llama V3.2 1B Instruct Text Generation"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 3.60
    max_context_length = 131072

class LlamaV32TextGeneration1BInstructQ80(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 1B instruct model with Q8-0 quantization.
    """
    task = "text-generation"
    model = "llama-v3-2-1b-instruct-q8-0"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-2-1b-instruct-q8-0.gguf"
    display_name = "Llama V3.2 1B Instruct Text Generation (Q8-0)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 2.43
    max_context_length = 131072

class LlamaV32TextGeneration1BInstructQ6K(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 1B instruct model with Q6-K quantization.
    """
    task = "text-generation"
    model = "llama-v3-2-1b-instruct-q6-k"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-2-1b-instruct-q6-k.gguf"
    display_name = "Llama V3.2 1B Instruct Text Generation (Q6-K)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 2.15
    max_context_length = 131072

class LlamaV32TextGeneration1BInstructQ5KM(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 1B instruct model with Q5-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-2-1b-instruct-q5-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-2-1b-instruct-q5-k-m.gguf"
    display_name = "Llama V3.2 1B Instruct Text Generation (Q5-K-M)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 2.02
    max_context_length = 131072

class LlamaV32TextGeneration1BInstructQ4KM(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 1B instruct model with Q4-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-2-1b-instruct-q4-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-2-1b-instruct-q4-k-m.gguf"
    display_name = "Llama V3.2 1B Instruct Text Generation (Q4-K-M)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 1.64
    max_context_length = 131072

class LlamaV32TextGeneration1BInstructQ3KL(Llama3TextGeneration):
    """
    Text generation using llama.cpp and llama-3 1B instruct model with Q3-K-M quantization.
    """
    task = "text-generation"
    model = "llama-v3-2-1b-instruct-q3-k-l"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-llama-v3-2-1b-instruct-q3-k-l.gguf"
    display_name = "Llama V3.2 1B Instruct Text Generation (Q3-K-L)"
    static_memory_gb = 4.21
    static_gpu_memory_gb = 1.58
    max_context_length = 131072
