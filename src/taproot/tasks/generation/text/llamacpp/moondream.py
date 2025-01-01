from typing import Optional, Any
from taproot.constants import *
from .base import LlamaTextGeneration, LlamaImageCaptioning

__all__ = [
    "MoondreamV2VisualQuestionAnswering",
    "MoondreamV2ImageCaptioning",
]

class MoondreamV2:
    """
    Text generation using llama.cpp and the Moondream V2 model.

    Note: Type definitions in this class have to match the parent class and can't be ommitted
    or else this file won't pass mypy checks.
    """
    """Global Task Metadata"""
    model: Optional[str] = "moondream-v2"
    default: bool = True
    static_memory_gb: Optional[float] = 0.472
    static_gpu_memory_gb: Optional[float] = 4.44

    """Author Metadata"""
    author: Optional[str] = "Vikhyat Korrapati"
    author_url: Optional[str] = "https://huggingface.co/vikhyatk/moondream2"
    author_journal: Optional[str] = "Hugging Face"
    author_journal_volume: Optional[str] = "10.57967/hf/3219"
    author_journal_year: Optional[int] = 2024
    author_journal_title: Optional[str] = "Moondream2"

    """License Metadata"""
    license: Optional[str] = LICENSE_APACHE

    """Task-Specific Metadata"""
    supports_image: bool = True
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/visual-question-answering-moondream-v2.fp16.gguf"
    mmproj_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-encoding-clip-moondream-v2-mmproj.fp16.gguf"
    chat_format: Optional[str] = "moondream2"

    def get_chat_handler(self) -> Any:
        """
        Get the chat handler for this model.
        """
        import logging
        from taproot.util import logger
        from llama_cpp.llama_chat_format import MoondreamChatHandler
        return MoondreamChatHandler(
            clip_model_path=self.get_model_file(self.mmproj_url), # type: ignore[attr-defined]
            verbose=logger.isEnabledFor(logging.DEBUG),
        )

class MoondreamV2VisualQuestionAnswering(MoondreamV2, LlamaTextGeneration):
    """
    Text generation using llama.cpp and the Moondream V2 model.
    """
    task: str = "visual-question-answering"
    display_name: Optional[str] = "Moondream V2 Visual Question Answering"

class MoondreamV2ImageCaptioning(MoondreamV2, LlamaImageCaptioning):
    """
    Image captioning using llama.cpp and the Moondream V2 model.
    """
    task: str = "image-captioning"
    component_tasks = {"llama": MoondreamV2VisualQuestionAnswering}
    display_name: Optional[str] = "Moondream V2 Image Captioning"
