from typing import Optional

from ...base import LlamaTextGeneration, LlamaImageCaptioning
from ..base import Llava

__all__ = [
    "LlavaV16ExtraLargeVisualQuestionAnsweringQ5KM",
    "LlavaV16ExtraLargeImageCaptioningQ5KM",
    "LlavaV16ExtraLargeVisualQuestionAnsweringQ4KM",
    "LlavaV16ExtraLargeImageCaptioningQ4KM",
    "LlavaV16ExtraLargeVisualQuestionAnsweringQ3KM",
    "LlavaV16ExtraLargeImageCaptioningQ3KM",
]

# Q5KM
    
class LlavaV16ExtraLargeQ5KM(Llava):
    """
    Text generation using llama.cpp and llava-1.6 34B instruction model using q5-k-m quantization.
    """
    model: Optional[str] = "llava-v1-6-34b-q5-k-m"
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-llava-v1-6-34b-q5-k-m.gguf"
    mmproj_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-encoding-clip-llava-mmproj-v1-6-34b.fp16.gguf"
    static_memory_gb: Optional[float] = .577
    static_gpu_memory_gb: Optional[float] = 24.96

class LlavaV16ExtraLargeVisualQuestionAnsweringQ5KM(LlavaV16ExtraLargeQ5KM, LlamaTextGeneration):
    """
    Text generation using llama.cpp and llava-1.6 34B instruction model using q5-k-m quantization.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "LLaVA V1.6 34B (Q5-K-M) Visual Question Answering"

class LlavaV16ExtraLargeImageCaptioningQ5KM(LlavaV16ExtraLargeQ5KM, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and llava-1.6 34B instruction model using q5-k-m quantization.
    """
    task: str = "image-captioning"
    display_name: Optional[str] = "LLaVA V1.6 34B (Q5-K-M) Image Captioning"
    component_tasks = {"llama": LlavaV16ExtraLargeVisualQuestionAnsweringQ5KM}

# Q4KM
    
class LlavaV16ExtraLargeQ4KM(Llava):
    """
    Text generation using llama.cpp and llava-1.6 34B instruction model using q4-k-m quantization.
    """
    model: Optional[str] = "llava-v1-6-34b-q4-k-m"
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-llava-v1-6-34b-q4-k-m.gguf"
    mmproj_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-encoding-clip-llava-mmproj-v1-6-34b.fp16.gguf"
    static_memory_gb: Optional[float] = .577
    static_gpu_memory_gb: Optional[float] = 21.88

class LlavaV16ExtraLargeVisualQuestionAnsweringQ4KM(LlavaV16ExtraLargeQ4KM, LlamaTextGeneration):
    """
    Text generation using llama.cpp and llava-1.6 34B instruction model using q4-k-m quantization.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "LLaVA V1.6 34B (Q4-K-M) Visual Question Answering"

class LlavaV16ExtraLargeImageCaptioningQ4KM(LlavaV16ExtraLargeQ4KM, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and llava-1.6 34B instruction model using q4-k-m quantization.
    """
    task: str = "image-captioning"
    display_name: Optional[str] = "LLaVA V1.6 34B (Q4-K-M) Image Captioning"
    component_tasks = {"llama": LlavaV16ExtraLargeVisualQuestionAnsweringQ4KM}

# Q3KM

class LlavaV16ExtraLargeQ3KM(Llava):
    """
    Text generation using llama.cpp and llava-1.6 34B instruction model using q3-k-m quantization.
    """
    model: Optional[str] = "llava-v1-6-34b-q3-k-m"
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-llava-v1-6-34b-q3-k-m.gguf"
    mmproj_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-encoding-clip-llava-mmproj-v1-6-34b.fp16.gguf"
    static_memory_gb: Optional[float] = .577
    static_gpu_memory_gb: Optional[float] = 18.06

class LlavaV16ExtraLargeVisualQuestionAnsweringQ3KM(LlavaV16ExtraLargeQ3KM, LlamaTextGeneration):
    """
    Text generation using llama.cpp and llava-1.6 34B instruction model using q3-k-m quantization.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "LLaVA V1.6 34B (Q3-K-M) Visual Question Answering"

class LlavaV16ExtraLargeImageCaptioningQ3KM(LlavaV16ExtraLargeQ3KM, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and llava-1.6 34B instruction model using q3-k-m quantization.
    """
    task: str = "image-captioning"
    display_name: Optional[str] = "LLaVA V1.6 34B (Q3-K-M) Image Captioning"
    component_tasks = {"llama": LlavaV16ExtraLargeVisualQuestionAnsweringQ3KM}
