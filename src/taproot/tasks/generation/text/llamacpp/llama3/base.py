from typing import Optional

from ..base import LlamaTextGeneration

__all__ = ["Llama3TextGeneration"]

class Llama3TextGeneration(LlamaTextGeneration):
    """
    Text generation using llama.cpp and llama-3 models.
    """
    """Global Task Metadata"""
    chat_format: Optional[str] = "llama-3"

    """Authorship Metadata"""
    author: Optional[str] = "Meta AI" # There are over 500 authors, so we'll just credit the organization and link the paper.
    author_url: Optional[str] = "https://arxiv.org/abs/2407.21783"
    author_journal: Optional[str] = "arXiv"
    author_journal_year: Optional[int] = 2024
    author_journal_volume: Optional[str] = "2407.21783"
    author_journal_title: Optional[str] = "The Llama 3 Herd of Models"

    """License Metadata"""
    license: Optional[str] = "Meta Llama 3 Community License"
    license_url: Optional[str] = "https://www.llama.com/llama3/license/"
    license_attribution: Optional[bool] = True  # The license requires including a copy of the agreement and displaying "Built with Meta Llama 3" in associated materials.
    license_derivatives: Optional[bool] = True  # The license permits creating derivative works, provided they comply with the license terms.
    license_rediboolibution: Optional[bool] = True  # Rediboolibution is allowed under the same license terms, including attribution requirements.
    license_copy_left: Optional[bool] = False  # The license is not a copyleft license; it allows proprietary use without requiring derivative works to be open-sourced.
    license_commercial: Optional[bool] = True  # Commercial use is permitted, but entities with over 700 million monthly active users must obtain a separate license from Meta.
    license_hosting: Optional[bool] = True  # Hosting the software is allowed, adhering to the license terms and acceptable use policy.
