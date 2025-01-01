from typing import Optional

from ...base import LlamaTextGeneration, LlamaImageCaptioning
from ..base import Llava

__all__ = [
    "LlavaV15VisualQuestionAnswering",
    "LlavaV15ImageCaptioning",
    "LlavaV15VisualQuestionAnsweringQ8",
    "LlavaV15ImageCaptioningQ8",
    "LlavaV15VisualQuestionAnsweringQ6K",
    "LlavaV15ImageCaptioningQ6K",
    "LlavaV15VisualQuestionAnsweringQ5KM",
    "LlavaV15ImageCaptioningQ5KM",
    "LlavaV15VisualQuestionAnsweringQ4KM",
    "LlavaV15ImageCaptioningQ4KM",
    "LlavaV15VisualQuestionAnsweringQ3KM",
    "LlavaV15ImageCaptioningQ3KM",
]

class LlavaV15(Llava):
    """
    Text generation using llama.cpp and llava-1.5 7B instruction model.
    """
    model: Optional[str] = "llava-v1-5-7b"
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-llava-v1-5-7b.fp16.gguf"
    mmproj_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-encoding-clip-llava-mmproj-v1-5-7b.fp16.gguf"
    static_memory_gb: Optional[float] = .577
    static_gpu_memory_gb: Optional[float] = 15.8

"""Fp16 tasks"""

class LlavaV15VisualQuestionAnswering(LlavaV15, LlamaTextGeneration):
    """
    Text generation using llama.cpp and llava-1.5 7B instruction model.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "LLaVA V1.5 7B Visual Question Answering"

class LlavaV15ImageCaptioning(LlavaV15, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and llava-1.5 7B instruction model.
    """
    task: str = "image-captioning"
    display_name: Optional[str] = "LLaVA V1.5 7B Image Captioning"
    component_tasks = {"llama": LlavaV15VisualQuestionAnswering}

"""Q8 tasks"""

class LlavaV15Q8(LlavaV15):
    """
    Text generation using llama.cpp and llava-1.5 7B instruction model using q8 quantization.
    """
    model: Optional[str] = "llava-v1-5-7b-q8"
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-llava-v1-5-7b-q8-0.gguf"
    static_memory_gb: Optional[float] = .577
    static_gpu_memory_gb: Optional[float] = 9.9

class LlavaV15VisualQuestionAnsweringQ8(LlavaV15Q8, LlamaTextGeneration):
    """
    Text generation using llama.cpp and llava-1.5 7B instruction model using q8 quantization.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "LLaVA V1.5 7B (Q8-0) Visual Question Answering"

class LlavaV15ImageCaptioningQ8(LlavaV15Q8, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and llava-1.5 7B instruction model using q8 quantization.
    """
    task: str = "image-captioning"
    display_name: Optional[str] = "LLaVA V1.5 7B (Q8-0) Image Captioning"
    component_tasks = {"llama": LlavaV15VisualQuestionAnsweringQ8}

"""Q6K tasks"""

class LlavaV15Q6K(LlavaV15):
    """
    Text generation using llama.cpp and llava-1.5 7B instruction model using q6k quantization.
    """
    model: Optional[str] = "llava-v1-5-7b-q6-k"
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-llava-v1-5-7b-q6-k.gguf"
    static_memory_gb: Optional[float] = .577
    static_gpu_memory_gb: Optional[float] = 8.4

class LlavaV15VisualQuestionAnsweringQ6K(LlavaV15Q6K, LlamaTextGeneration):
    """
    Text generation using llama.cpp and llava-1.5 7B instruction model using q6k quantization.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "LLaVA V1.5 7B (Q6-K) Visual Question Answering"

class LlavaV15ImageCaptioningQ6K(LlavaV15Q6K, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and llava-1.5 7B instruction model using q6k quantization.
    """
    task: str = "image-captioning"
    component_tasks = {"llama": LlavaV15VisualQuestionAnsweringQ6K}
    display_name: Optional[str] = "LLaVA V1.5 7B (Q6-K) Image Captioning"

"""Q5KM tasks"""

class LlavaV15Q5KM(LlavaV15):
    """
    Text generation using llama.cpp and llava-1.5 7B instruction model using q5-k-m quantization.
    """
    model: Optional[str] = "llava-v1-5-7b-q5-k-m"
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-llava-v1-5-7b-q5-k-m.gguf"
    static_memory_gb: Optional[float] = .577
    static_gpu_memory_gb: Optional[float] = 7.71

class LlavaV15VisualQuestionAnsweringQ5KM(LlavaV15Q5KM, LlamaTextGeneration):
    """
    Text generation using llama.cpp and llava-1.5 7B instruction model using q5-k-m quantization.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "LLaVA V1.5 7B (Q5-K-M) Visual Question Answering"

class LlavaV15ImageCaptioningQ5KM(LlavaV15Q5KM, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and llava-1.5 7B instruction model using q5-k-m quantization.
    """
    task: str = "image-captioning"
    component_tasks = {"llama": LlavaV15VisualQuestionAnsweringQ5KM}
    display_name: Optional[str] = "LLaVA V1.5 7B (Q5-K-M) Image Captioning"

"""Q4KM tasks"""

class LlavaV15Q4KM(LlavaV15):
    """
    Text generation using llama.cpp and llava-1.5 7B instruction model using q4-k-m quantization.
    """
    model: Optional[str] = "llava-v1-5-7b-q4-k-m"
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-llava-v1-5-7b-q4-k-m.gguf"
    static_memory_gb: Optional[float] = .577
    static_gpu_memory_gb: Optional[float] = 7.04

class LlavaV15VisualQuestionAnsweringQ4KM(LlavaV15Q4KM, LlamaTextGeneration):
    """
    Text generation using llama.cpp and llava-1.5 7B instruction model using q4-k-m quantization.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "LLaVA V1.5 7B (Q4-K-M) Visual Question Answering"

class LlavaV15ImageCaptioningQ4KM(LlavaV15Q4KM, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and llava-1.5 7B instruction model using q4-k-m quantization.
    """
    task: str = "image-captioning"
    display_name: Optional[str] = "LLaVA V1.5 7B (Q4-K-M) Image Captioning"
    component_tasks = {"llama": LlavaV15VisualQuestionAnsweringQ4KM}

"""Q3KM tasks"""

class LlavaV15Q3KM(LlavaV15):
    """
    Text generation using llama.cpp and llava-1.5 7B instruction model using q3-k-m quantization.
    """
    model: Optional[str] = "llava-v1-5-7b-q3-k-m"
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-llava-v1-5-7b-q3-k-m.gguf"
    static_memory_gb: Optional[float] = .577
    static_gpu_memory_gb: Optional[float] = 6.33

class LlavaV15VisualQuestionAnsweringQ3KM(LlavaV15Q3KM, LlamaTextGeneration):
    """
    Text generation using llama.cpp and llava-1.5 7B instruction model using q3-k-m quantization.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "LLaVA V1.5 7B (Q3-K-M) Visual Question Answering"

class LlavaV15ImageCaptioningQ3KM(LlavaV15Q3KM, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and llava-1.5 7B instruction model using q3-k-m quantization.
    """
    task: str = "image-captioning"
    display_name: Optional[str] = "LLaVA V1.5 7B (Q3-K-M) Image Captioning"
    component_tasks = {"llama": LlavaV15VisualQuestionAnsweringQ3KM}
