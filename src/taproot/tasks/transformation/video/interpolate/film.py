from .base import VideoFrameInterpolationTaskBase
from ...image.interpolate.film import FILMInterpolation

__all__ = ["FILMVideoInterpolation"]

class FILMVideoInterpolation(VideoFrameInterpolationTaskBase):
    """
    Video Interpolation with FiLM (Frame Interpolation for Large Motion)
    """
    """Global Task Metadata"""
    task = "video-interpolation"
    model = "film"
    default = True
    display_name = "FiLM Video Interpolation"
    component_tasks = {"interpolate": FILMInterpolation}

    """Authorship Metadata"""
    author = FILMInterpolation.author
    author_additional = FILMInterpolation.author_additional
    author_url = FILMInterpolation.author_url
    author_journal = FILMInterpolation.author_journal
    author_journal_year = FILMInterpolation.author_journal_year
    author_journal_title = FILMInterpolation.author_journal_title
    author_affiliations = FILMInterpolation.author_affiliations

    """License Metadata"""
    license = FILMInterpolation.license # Apache
