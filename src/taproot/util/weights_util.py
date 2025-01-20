from __future__ import annotations

from typing import Callable, Optional, Union, List

from ..constants import *

from .attribution_util import AttributionMixin
from .discovery_util import NamedDiscoveryMixin
from .download_util import check_download_files_to_dir

__all__ = ["PretrainedWeights"]

class PretrainedWeights(AttributionMixin, NamedDiscoveryMixin):
    """
    A class to represent a hosted weight
    """
    url: Union[str, List[str]]

    @classmethod
    def get_files(
        cls,
        weight_dir: str=DEFAULT_MODEL_DIR,
        download_chunk_size: int=8192,
        check_size: bool=False,
        progress_callback: Optional[Callable[[int, int, int, int], None]]=None,
        text_callback: Optional[Callable[[str], None]]=None,
        authorization: Optional[str]=None,
    ) -> List[str]:
        """
        Download the weight files.
        """
        return check_download_files_to_dir(
            [cls.url] if isinstance(cls.url, str) else cls.url,
            weight_dir,
            chunk_size=download_chunk_size,
            check_size=check_size,
            progress_callback=progress_callback,
            text_callback=text_callback,
            authorization=authorization
        )
