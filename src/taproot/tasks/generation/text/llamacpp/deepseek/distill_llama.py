from .r1 import DeepSeekR1Llama3TextGeneration

__all__ = [
    "DeepSeekR1Llama3TextGeneration8B",
    "DeepSeekR1Llama3TextGeneration8BQ80",
    "DeepSeekR1Llama3TextGeneration8BQ6K",
    "DeepSeekR1Llama3TextGeneration8BQ5KM",
    "DeepSeekR1Llama3TextGeneration8BQ4KM",
    "DeepSeekR1Llama3TextGeneration8BQ3KM",
    "DeepSeekR1Llama3TextGeneration8BQ2K",
]

class DeepSeekR1Llama3TextGeneration8B(DeepSeekR1Llama3TextGeneration):
    task = "text-generation"
    model = "deepseek-r1-llama-8b"
    default = False
    model_url = "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-f16.gguf?filename=text-generation-deepseek-r1-llama-8b-fp16.gguf"
    max_context_length = 131072
    static_gpu_memory_gb = 16.2

class DeepSeekR1Llama3TextGeneration8BQ80(DeepSeekR1Llama3TextGeneration):
    task = "text-generation"
    model = "deepseek-r1-llama-8b-q8-0"
    default = False
    model_url = "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf?filename=text-generation-deepseek-r1-llama-8b-q8-0.gguf"
    static_gpu_memory_gb = 9.45
    max_context_length = 131072

class DeepSeekR1Llama3TextGeneration8BQ6K(DeepSeekR1Llama3TextGeneration):
    task = "text-generation"
    model = "deepseek-r1-llama-8b-q6-k"
    default = False
    model_url = "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf?filename=text-generation-deepseek-r1-llama-8b-q6-k.gguf"
    static_gpu_memory_gb = 7.73
    max_context_length = 131072

class DeepSeekR1Llama3TextGeneration8BQ5KM(DeepSeekR1Llama3TextGeneration):
    task = "text-generation"
    model = "deepseek-r1-llama-8b-q5-k-m"
    default = False
    model_url = "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf?filename=text-generation-deepseek-r1-llama-8b-q5-k-m.gguf"
    static_gpu_memory_gb = 6.96
    max_context_length = 131072

class DeepSeekR1Llama3TextGeneration8BQ4KM(DeepSeekR1Llama3TextGeneration):
    task = "text-generation"
    model = "deepseek-r1-llama-8b-q4-k-m"
    default = False
    model_url = "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf?filename=text-generation-deepseek-r1-llama-8b-q4-k-m.gguf"
    static_gpu_memory_gb = 6.24
    max_context_length = 131072

class DeepSeekR1Llama3TextGeneration8BQ3KM(DeepSeekR1Llama3TextGeneration):
    task = "text-generation"
    model = "deepseek-r1-llama-8b-q3-k-m"
    default = False
    model_url = "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q3_K_M.gguf?filename=text-generation-deepseek-r1-llama-8b-q3-k-m.gguf"
    static_gpu_memory_gb = 5.44
    max_context_length = 131072

class DeepSeekR1Llama3TextGeneration8BQ2K(DeepSeekR1Llama3TextGeneration):
    task = "text-generation"
    model = "deepseek-r1-llama-8b-q2-k"
    default = False
    model_url = "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf?filename=text-generation-deepseek-r1-llama-8b-q2-k.gguf"
    static_gpu_memory_gb = 4.71
    max_context_length = 131072
