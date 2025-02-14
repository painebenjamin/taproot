from __future__ import annotations

from typing import Union, Tuple, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = [
    "tile_image",
    "image_tiles",
]

def tile_image(image: Image, tiles: Union[int, Tuple[int, int]]) -> Image:
    """
    Given an image and number of tiles, create a tiled image.
    Accepts either an integer (squre tiles) or tuple (rectangular)
    """
    from PIL import Image
    width, height = image.size
    if isinstance(tiles, tuple):
        width_tiles, height_tiles = tiles
    else:
        width_tiles, height_tiles = tiles, tiles
    tiled = Image.new(image.mode, (width * width_tiles, height * height_tiles))
    for i in range(width_tiles):
        for j in range(height_tiles):
            tiled.paste(image, (i * width, j * height))
    return tiled

def image_tiles(
    image: Image,
    tile_size: Union[int, Tuple[int, int]],
    tile_stride: Union[int, Tuple[int, int]],
) -> Iterator[Image]:
    """
    Gets image tiles using sliding windows.
    """
    from ..misc_util import sliding_2d_windows
    width, height = image.size
    for x0, x1, y0, y1 in sliding_2d_windows(width, height, tile_size, tile_stride):
        cropped = image.crop((x0, y0, x1, y1))
        setattr(cropped, "coordinates", (x0, y0, x1, y1))
        yield cropped
