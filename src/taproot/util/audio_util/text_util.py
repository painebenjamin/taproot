# Utilities for processing text for audio generation

__all__ = ["get_punctuation_pause_ratio"]

def get_punctuation_pause_ratio(text: str) -> float:
    """
    Check if the text ends with punctuation and return the pause duration ratio.

    >>> get_punctuation_pause_ratio("Hello, world!")
    1.0
    >>> get_punctuation_pause_ratio("Hello,")
    0.5
    >>> get_punctuation_pause_ratio("Hello")
    0.0

    :param text: The text to check.
    :return: The pause duration ratio
    """
    char = text.strip()[-1]
    if char in [".", "!", "?", "。", "，", "！", "？"]: 
        return 1.0 
    if char in [",", ";", "；"]:
        return 0.5 
    return 0.0 
