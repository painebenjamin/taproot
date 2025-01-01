from typing import Optional

from ...base import LlamaTextGeneration, LlamaImageCaptioning
from ..base import Llava

__all__ = [
    "LlavaV15LargeVisualQuestionAnsweringQ8",
    "LlavaV15LargeImageCaptioningQ8",
    "LlavaV15LargeVisualQuestionAnsweringQ6K",
    "LlavaV15LargeImageCaptioningQ6K",
    "LlavaV15LargeVisualQuestionAnsweringQ5KM",
    "LlavaV15LargeImageCaptioningQ5KM",
    "LlavaV15LargeVisualQuestionAnsweringQ4",
    "LlavaV15LargeImageCaptioningQ4",
]

"""Q8 tasks"""

class LlavaV15LargeQ8(Llava):
    """
    Text generation using llama.cpp and llava-1.5 13B instruction model with q8 quantization.
    """
    model: Optional[str] = "llava-v1-5-13b"
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-llava-v1-5-13b-q8-0.gguf"
    mmproj_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-encoding-clip-llava-mmproj-v1-5-13b.fp16.gguf"
    static_memory_gb: Optional[float] = .577
    static_gpu_memory_gb: Optional[float] = 17.51

class LlavaV15LargeVisualQuestionAnsweringQ8(LlavaV15LargeQ8, LlamaTextGeneration):
    """
    Text generation using llama.cpp and llava-1.5 13B instruction model.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "LLaVA V1.51 13B (Q8-0) Visual Question Answering"

class LlavaV15LargeImageCaptioningQ8(LlavaV15LargeQ8, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and llava-1.5 13B instruction model.
    """
    task: str = "image-captioning"
    display_name: Optional[str] = "LLaVA V1.51 13B (Q8-0) Image Captioning"
    component_tasks = {"llama": LlavaV15LargeVisualQuestionAnsweringQ8}

"""Q6-K tasks"""

class LlavaV15LargeQ6K(LlavaV15LargeQ8):
    """
    Text generation using llama.cpp and llava-1.5 13B instruction model with q6-k quantization.
    """
    model: Optional[str] = "llava-v1-5-13b-q6-k"
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-llava-v1-5-13b-q6-k.gguf"
    static_memory_gb: Optional[float] = .577
    static_gpu_memory_gb: Optional[float] = 14.54

class LlavaV15LargeVisualQuestionAnsweringQ6K(LlavaV15LargeQ6K, LlamaTextGeneration):
    """
    Text generation using llama.cpp and llava-1.5 13B instruction model.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "LLaVA V1.51 13B (Q6-K) Visual Question Answering"

class LlavaV15LargeImageCaptioningQ6K(LlavaV15LargeQ6K, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and llava-1.5 13B instruction model.
    """
    task: str = "image-captioning"
    display_name: Optional[str] = "LLaVA V1.51 13B (Q6-K) Image Captioning"
    component_tasks = {"llama": LlavaV15LargeVisualQuestionAnsweringQ6K}

"""Q5-K-M tasks"""

class LlavaV15LargeQ5KM(LlavaV15LargeQ8):
    """
    Text generation using llama.cpp and llava-1.5 13B instruction model with q5-k-m quantization.
    """
    model: Optional[str] = "llava-v1-5-13b-q5-k-m"
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-llava-v1-5-13b-q5-k-m.gguf"
    static_memory_gb: Optional[float] = .577
    static_gpu_memory_gb: Optional[float] = 13.17

class LlavaV15LargeVisualQuestionAnsweringQ5KM(LlavaV15LargeQ5KM, LlamaTextGeneration):
    """
    Text generation using llama.cpp and llava-1.5 13B instruction model.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "LLaVA V1.51 13B (Q5-K-M) Visual Question Answering"

class LlavaV15LargeImageCaptioningQ5KM(LlavaV15LargeQ5KM, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and llava-1.5 13B instruction model.
    """
    task: str = "image-captioning"
    display_name: Optional[str] = "LLaVA V1.51 13B (Q5-K-M) Image Captioning"
    component_tasks = {"llama": LlavaV15LargeVisualQuestionAnsweringQ5KM}

"""Q4-0 tasks"""

class LlavaV15LargeQ4(LlavaV15LargeQ8):
    """
    Text generation using llama.cpp and llava-1.5 13B instruction model with q4 quantization.
    """
    model: Optional[str] = "llava-v1-5-13b-q4-0"
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-llava-v1-5-13b-q4-0.gguf"
    static_memory_gb: Optional[float] = .577
    static_gpu_memory_gb: Optional[float] = 11.48

class LlavaV15LargeVisualQuestionAnsweringQ4(LlavaV15LargeQ4, LlamaTextGeneration):
    """
    Text generation using llama.cpp and llava-1.5 13B instruction model.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "LLaVA V1.51 13B (Q4-0) Visual Question Answering"

class LlavaV15LargeImageCaptioningQ4(LlavaV15LargeQ4, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and llava-1.5 13B instruction model.
    """
    task: str = "image-captioning"
    display_name: Optional[str] = "LLaVA V1.51 13B (Q4-0) Image Captioning"
    component_tasks = {"llama": LlavaV15LargeVisualQuestionAnsweringQ4}
