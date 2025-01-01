from typing import Union, Tuple, Literal, List, Optional, Sequence
from typing_extensions import TypedDict, NotRequired
from PIL import Image
from numpy.typing import ArrayLike
from torch import Tensor

from .util import SeedType # re-exported

__all__ = [
    "SingleImageType",
    "ImageType",
    "ImageResultType",
    "SingleAudioType",
    "AudioType",
    "AudioResultType",
    "TextMessageDict",
    "MessageDict",
    "PromptType",
    "SeedType",
]

SingleImageType = Union[
    str, Image.Image, ArrayLike, Tensor
]
ImageType = Union[
    SingleImageType,
    Sequence[SingleImageType],
]
ImageResultType = Union[
    SingleImageType,
    Sequence[SingleImageType],
    Sequence[Tuple[SingleImageType, ...]],
]
SingleAudioType = Union[
    str, bytes, bytearray, ArrayLike, Tensor, Sequence[Tuple[Union[int, float], ...]]
]
AudioType = Union[
    SingleAudioType,
    Sequence[SingleAudioType],
]
AudioResultType = Union[
    SingleAudioType,
    Sequence[SingleAudioType],
    Sequence[Tuple[SingleAudioType, ...]],
]
class MessageDict(TypedDict):
    text: str
    role: NotRequired[Literal["user", "assistant", "system"]]
    image: NotRequired[Optional[SingleImageType]]

PromptType = Union[
    str, # one text prompt
    List[str], # conversation as [user, assistant, user, assistant, ...]
    MessageDict, # one message with optional image
    List[MessageDict], # conversation with optional images
]
SeedType = SeedType # Silence importchecker
