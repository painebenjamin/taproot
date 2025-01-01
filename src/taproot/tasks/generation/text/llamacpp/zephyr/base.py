import re

from taproot.constants import *
from ..base import LlamaTextGeneration

__all__ = [
    "Zephyr7BTextGeneration",
]

ZEPHYR_RESPONSE_REGEX = re.compile(r"^[\"](.*)[\"] \(\d+ characters\)$")

class Zephyr7BTextGeneration(LlamaTextGeneration):
    """
    Zephyr 7B Text Generation (parent class)
    """

    """Authorship Metadata"""
    author = "Lewis Tunstall"
    author_url = "https://arxiv.org/abs/2310.16944"
    author_additional = ["Edward Beeching", "Nathan Lambert", "Nazneen Rajani", "Kashif Rasul", "Younes Belkada", "Shengyi Huang", "Leandro von Werra", "Clémentine Fourrier", "Nathan Habib", "Nathan Sarrazin", "Omar Sansevier", "Alexander M. Rush", "Thomas Wolf"]
    author_journal = "arXiv"
    author_journal_year = 2023
    author_journal_volume = "2310.16944"
    author_journal_title = "Zephyr: Direct Distillation of LM Alignment"

    """License Metadata"""
    license = LICENSE_MIT

    """Task-Specific Metadata"""
    chat_format = "zephyr"

    def trim_response(self, text: str) -> str:
        """
        Trims the response to remove any unwanted text.
        """
        match = ZEPHYR_RESPONSE_REGEX.match(text)
        if match:
            return super().trim_response(match.group(1))
        return super().trim_response(text)
