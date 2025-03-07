from .base import VideoFrameInterpolationTaskBase
from ...image.interpolate.rife import RIFEImageInterpolation

__all__ = ["RIFEVideoInterpolation"]

class RIFEVideoInterpolation(VideoFrameInterpolationTaskBase):
    """
    Video Interpolation with RIFE (Real-Time Intermediate Flow Estimation)
    """
    """Global Task Metadata"""
    task = "video-interpolation"
    model = "rife"
    display_name = "RIFE Video Interpolation"
    component_tasks = {"interpolate": RIFEImageInterpolation}

    """Authorship Metadata"""
    author = RIFEImageInterpolation.author
    author_additional = RIFEImageInterpolation.author_additional
    author_url = RIFEImageInterpolation.author_url
    author_journal = RIFEImageInterpolation.author_journal
    author_journal_year = RIFEImageInterpolation.author_journal_year
    author_journal_title = RIFEImageInterpolation.author_journal_title
    author_affiliations = RIFEImageInterpolation.author_affiliations

    """License Metadata"""
    license = RIFEImageInterpolation.license # Apache
