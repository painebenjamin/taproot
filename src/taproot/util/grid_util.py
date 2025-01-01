from __future__ import annotations

import textwrap

from typing import (
    List,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING
)

from dataclasses import dataclass

from taproot.constants import *
from taproot.util.image_util import fit_image, to_pil_array

if TYPE_CHECKING:
    from taproot.hinting import ImageType
    from PIL.Image import Image
    from PIL.ImageFont import FreeTypeFont

__all__ = [
    "GridMaker",
    "make_grid"
]

@dataclass
class GridMaker:
    """
    A small class for building grids of images.
    """

    use_video: bool = False
    font_url: str = "https://github.com/adobe-fonts/source-code-pro/raw/release/TTF/SourceCodePro-Regular.ttf"
    font_size: int = 12
    num_columns: int = 4
    image_size: Union[int, Tuple[int, int]] = 256
    image_fit: IMAGE_FIT_LITERAL = "contain"
    image_anchor: IMAGE_ANCHOR_LITERAL = "center-center"
    caption_color: IMAGE_CAPTION_COLOR_LITERAL = "white-on-black"
    caption_type: IMAGE_CAPTION_TYPE_LITERAL = "separate" # or 'overlay'
    caption_position: IMAGE_CAPTION_POSITION_LITERAL = "bottom"
    caption_align: IMAGE_CAPTION_ALIGN_LITERAL = "center"
    caption_height: int = 50

    @property
    def font(self) -> FreeTypeFont:
        """
        Gets the default system font.
        """
        if not hasattr(self, "_font"):
            from PIL import ImageFont
            from requests import get
            from io import BytesIO
            self._font = ImageFont.truetype(BytesIO(get(self.font_url).content), self.font_size)
        return self._font

    @property
    def text_max_length(self) -> int:
        """
        Calculates the maximum length of text
        """
        return (self.font_size - 10) + self.image_width // (self.font_size - 4)

    @property
    def image_width(self) -> int:
        """
        Gets the width of the image
        """
        return self.image_size if isinstance(self.image_size, int) else self.image_size[0]

    @property
    def image_height(self) -> int:
        """
        Gets the height of the image
        """
        return self.image_size if isinstance(self.image_size, int) else self.image_size[1]

    @property
    def background_color(self) -> Tuple[int, int, int]:
        """
        Gets the background color
        """
        if self.caption_color == "white-on-black":
            return (0, 0, 0)
        else:
            return (255, 255, 255)

    @property
    def text_color(self) -> Tuple[int, int, int]:
        """
        Gets the text color
        """
        if self.caption_color == "white-on-black":
            return (255, 255, 255)
        else:
            return (0, 0, 0)

    def get_wrapped_text(self, text: str) -> str:
        """
        Wraps text to fit the image width
        """
        return "\n".join(textwrap.wrap(text, width=self.text_max_length))

    def __call__(
        self,
        *images: Union[ImageType, Tuple[ImageType, str]],
    ) -> Union[Image, List[Image]]:
        """
        Builds the results into a collage.
        """
        from PIL import Image, ImageDraw
        # Homogenize input
        image_captions: List[Tuple[List[Image.Image], str]] = [
            (
                to_pil_array(image_input[0] if isinstance(image_input, tuple) else image_input),
                "" if not isinstance(image_input, tuple) else image_input[1]
            )
            for image_input in images
        ]

        # Get total images
        if self.use_video:
            total_images = len(image_captions)
        else:
            total_images = sum([len(images) for images, caption in image_captions])

        if total_images == 0:
            raise RuntimeError("No images passed.")

        # Get the number of rows and columns
        rows = total_images // self.num_columns
        if total_images % self.num_columns != 0:
            rows += 1

        columns = total_images % self.num_columns if total_images < self.num_columns else self.num_columns

        # Calculate image height based on rows and columns
        width = self.image_width * columns
        height = self.image_height * rows

        if self.caption_type == "separate":
            height += self.caption_height * rows

        # Create blank image
        grid = Image.new("RGB", (width, height), self.background_color)

        if self.use_video:
            frame_count = max([len(images) for images, caption in image_captions])
            # Repeat frames if necessary
            image_captions = [
                (
                    images + ([images[-1]] * (frame_count - len(images))),
                    caption
                )
                for images, caption in image_captions
            ]
            # Create a grid for each frame
            grid = [grid.copy() for i in range(frame_count)] # type: ignore[assignment]
            draw = [ImageDraw.Draw(image) for image in grid] # type: ignore[attr-defined]
        else:
            draw = ImageDraw.Draw(grid) # type: ignore[assignment]

        # Iterate through each result image and paste
        row, column = 0, 0
        for i, (images, caption) in enumerate(image_captions): # type: ignore[assignment]
            # Fit the image to the grid size
            image_width, image_height = images[0].size # type: ignore[union-attr,misc]
            images = fit_image( # type: ignore[assignment]
                images, # type: ignore[arg-type]
                width=self.image_width,
                height=self.image_height,
                fit=self.image_fit,
                anchor=self.image_anchor
            )

            for j, image in enumerate(images):
                # Figure out which image/draw to use
                if self.use_video:
                    target_image = grid[j] # type: ignore[index]
                    target_draw = draw[j]
                else:
                    target_image = grid
                    target_draw = draw # type: ignore[assignment]

                grid_x = column * self.image_width
                grid_y = row * (self.image_height + (self.caption_height if self.caption_type == "separate" else 0))

                if self.caption_type == "separate" and self.caption_position == "top":
                    grid_y += self.caption_height

                # Paste the image on the grid
                target_image.paste(image, (grid_x, grid_y))

                # Determine caption position
                wrapped_caption = self.get_wrapped_text(caption)
                _, _, text_width, text_height = target_draw.textbbox(
                    (0, 0),
                    wrapped_caption,
                    font=self.font,
                    align=self.caption_align
                )

                pad_x, pad_y = (2, 5)
                text_x, text_y = grid_x + pad_x, grid_y + pad_y

                if self.caption_type == "overlay":
                    if self.caption_position == "center":
                        text_y = grid_y + int((self.image_height - text_height) // 2) - pad_y
                    elif self.caption_position == "bottom":
                        text_y = grid_y + self.image_height - text_height - pad_y

                    if self.caption_align == "center":
                        text_x = grid_x + int((self.image_width - text_width) // 2)
                    elif self.caption_align == "right":
                        text_x = grid_x + self.image_width - text_width - pad_x

                elif self.caption_position == "top":
                    text_y = grid_y - self.caption_height + pad_y
                else:
                    text_y = grid_y + self.image_height + pad_y

                target_draw.multiline_text(
                    (text_x, text_y),
                    wrapped_caption,
                    fill=self.text_color,
                    font=self.font,
                    align=self.caption_align
                )

                # Increment as necessary
                if not self.use_video:
                    column += 1
                    if column >= self.num_columns:
                        row += 1
                        column = 0

            # Increment as necessary
            if self.use_video:
                column += 1
                if column >= self.num_columns:
                    row += 1
                    column = 0

        return grid

def make_grid(
    images: Sequence[Union[ImageType, Tuple[ImageType, str]]],
    use_video: bool = False,
    font_url: str = "https://github.com/adobe-fonts/source-code-pro/raw/release/TTF/SourceCodePro-Regular.ttf",
    font_size: int = 12,
    num_columns: int = 4,
    image_size: Union[int, Tuple[int, int]] = 256,
    image_fit: IMAGE_FIT_LITERAL = "contain",
    image_anchor: IMAGE_ANCHOR_LITERAL = "center-center",
    caption_color: IMAGE_CAPTION_COLOR_LITERAL = "white-on-black",
    caption_type: IMAGE_CAPTION_TYPE_LITERAL = "separate", # or 'overlay'
    caption_position: IMAGE_CAPTION_POSITION_LITERAL = "bottom",
    caption_align: IMAGE_CAPTION_ALIGN_LITERAL = "center",
    caption_height: int = 50,
) -> Union[Image, List[Image]]:
    """
    Builds the results into a collage.
    """
    grid_maker = GridMaker(
        use_video=use_video,
        font_url=font_url,
        font_size=font_size,
        num_columns=num_columns,
        image_size=image_size,
        image_fit=image_fit,
        image_anchor=image_anchor,
        caption_color=caption_color,
        caption_type=caption_type,
        caption_position=caption_position,
        caption_align=caption_align,
        caption_height=caption_height
    )
    return grid_maker(*images)
