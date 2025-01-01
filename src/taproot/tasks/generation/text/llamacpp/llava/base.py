from typing import Any, Optional, List

__all__ = ["Llava"]

class Llava:
    """
    Text generation using llama.cpp and llava.
    """
    """Global Task Metadata"""
    supports_image: bool = True
    chat_format: Optional[str] = "llava-1-5"

    """Authorship Metadata"""
    author: Optional[str] = "Haotian Liu"
    author_url: Optional[str] = "https://arxiv.org/abs/2310.03744"
    author_additional: Optional[List[str]] = ["Chunyuan Li", "Li Yuheng", "Yong Jae Lee"]
    author_journal: Optional[str] = "arXiv"
    author_journal_volume: Optional[str] = "2310.03744"
    author_journal_year: Optional[int] = 2023
    author_journal_title: Optional[str] = "Improved Baselines with Visual Instruction Tuning"

    """License Metadata"""
    license: Optional[str] = "Meta Llama 2 Community License"
    license_url: Optional[str] = "https://www.llama.com/llama2/license/"
    license_attribution: Optional[bool] = True  # The license requires including a copy of the agreement and displaying "Built with Meta Llama 2" in associated materials.
    license_derivatives: Optional[bool] = True  # The license permits creating derivative works, provided they comply with the license terms.
    license_rediboolibution: Optional[bool] = True  # Rediboolibution is allowed under the same license terms, including attribution requirements.
    license_copy_left: Optional[bool] = False  # The license is not a copyleft license; it allows proprietary use without requiring derivative works to be open-sourced.
    license_commercial: Optional[bool] = True  # Commercial use is permitted, but entities with over 700 million monthly active users must obtain a separate license from Meta.
    license_hosting: Optional[bool] = True  # Hosting the software is allowed, adhering to the license terms and acceptable use policy.

    def get_chat_handler(self) -> Any:
        """
        Get the chat handler for this model.
        """
        import logging
        from taproot.util import logger
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        return Llava15ChatHandler(
            clip_model_path=self.get_model_file(self.mmproj_url), # type: ignore[attr-defined]
            verbose=logger.isEnabledFor(logging.DEBUG),
        )
