from __future__ import annotations

from typing import Callable, Optional, Union, List, Type, Dict

from ..constants import *

from .attribution_util import AttributionMixin
from .download_util import check_download_files_to_dir

__all__ = ["HostedWeights"]

class HostedWeights(AttributionMixin):
    """
    A class to represent a hosted weight
    """
    name: str
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

    @classmethod
    def enumerate(cls) -> List[Type[HostedWeights]]:
        """
        Return the list of hosted weights.
        """
        classes = []
        if getattr(cls, "name", None) is not None:
            classes.append(cls)

        for subclass in cls.__subclasses__():
            classes.extend(subclass.enumerate())

        return classes

    @classmethod
    def get(cls, name: str) -> Optional[Type[HostedWeights]]:
        """
        Get a hosted weight by name.
        """
        for weight in cls.enumerate():
            if weight.name == name:
                return weight
        return None

    @classmethod
    def catalog(cls) -> Dict[str, Type[HostedWeights]]:
        """
        Return a dictionary of hosted weights.
        """
        return {weight.name: weight for weight in cls.enumerate()}
