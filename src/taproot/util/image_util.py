from __future__ import annotations

import io
import os
import math
import base64

from typing import Optional, Union, List, Tuple, Dict, Any, Iterator, TYPE_CHECKING

from ..constants import *
from .string_util import get_uuid
from .introspection_util import is_numpy_array, is_torch_tensor, is_pil_image

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image
    from torch import Tensor, dtype as TorchDType
    from turbojpeg import TurboJPEG # type: ignore[import-not-found,import-untyped,unused-ignore]
    from ..hinting import ImageType

__all__ = [
    "fit_image",
    "tile_image",
    "image_tiles",
    "image_from_uri",
    "images_are_equal",
    "get_frames_or_image",
    "get_frames_or_image_from_file",
    "save_frames_or_image",
    "create_mask",
    "scale_image",
    "get_image_metadata",
    "redact_images_from_metadata",
    "dilate_erode",
    "image_to_tensor",
    "tensor_to_image",
    "to_pil_array",
    "to_jpeg_array",
    "to_bhwc_ndarray",
    "to_bchw_tensor",
    "is_multi_image",
    "get_image_width_height",
    "serialize_image",
    "pad_image",
    "pad_image_to_nearest",
    "show_image",
    "EncodedImageProxy",
]

class EncodedImageProxy:
    """
    A proxy to an image that is already encoded in a specific format.
    This avoids re-encoding the image when it is transferred between
    different systems, as PIL images will always decode the data when
    initializing.

    Only the data is required, as the format and size can be inferred
    and the image can be decoded when necessary.
    """
    def __init__(
        self,
        data: bytes,
        format: Optional[str]=None,
        size: Optional[Tuple[int, int]]=None
    ) -> None:
        self.data = data
        self.passed_format = format
        self.passed_size = size

    @property
    def image(self) -> Image:
        """
        Decodes the image from the data.
        """
        if not hasattr(self, "_image"):
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = 2**32
            self._image = Image.open(io.BytesIO(self.data))
        return self._image

    @property
    def format(self) -> Optional[str]:
        """
        Returns the format of the image.
        Either the passed format or the format of the image.
        """
        if self.passed_format is not None:
            return self.passed_format
        return self.image.format

    @property
    def size(self) -> Tuple[int, int]:
        """
        Returns the size of the image.
        Either the passed size or the size of the image.
        """
        if self.passed_size is not None:
            return self.passed_size
        return self.image.size

    def __getattr__(self, attr: str) -> Any:
        """
        Passes all other attributes to the image.
        """
        if attr.startswith("_"):
            return super().__getattr__(attr) # type: ignore[misc]
        return getattr(self.image, attr)

def fit_image(
    image: Union[Image, List[Image]],
    width: int,
    height: int,
    fit: Optional[IMAGE_FIT_LITERAL] = None,
    anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
    offset_left: Optional[int] = None,
    offset_top: Optional[int] = None
) -> Union[Image, List[Image]]:
    """
    Given an image of unknown size, make it a known size with optional fit parameters.
    """
    if not isinstance(image, list):
        if getattr(image, "n_frames", 1) > 1:
            frames = []
            for i in range(image.n_frames): # type: ignore[attr-defined,unused-ignore]
                image.seek(i)
                frames.append(image.copy().convert("RGBA"))
            image = frames
    if isinstance(image, list):
        return [
            fit_image( # type: ignore[misc]
                img,
                width=width,
                height=height,
                fit=fit,
                anchor=anchor,
                offset_left=offset_left,
                offset_top=offset_top,
            )
            for img in image
        ]

    from PIL import Image

    if fit is None or fit == "actual":
        left, top = 0, 0
        crop_left, crop_top = 0, 0
        image_width, image_height = image.size

        if anchor is not None:
            top_part, left_part = anchor.split("-")

            if top_part == "center":
                top = height // 2 - image_height // 2
            elif top_part == "bottom":
                top = height - image_height

            if left_part == "center":
                left = width // 2 - image_width // 2
            elif left_part == "right":
                left = width - image_width

        blank_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        if offset_top is not None:
            top += offset_top
        if offset_left is not None:
            left += offset_left
        if image.mode == "RGBA":
            blank_image.paste(image, (left, top), image)
        else:
            blank_image.paste(image, (left, top))

        return blank_image

    elif fit == "contain":
        image_width, image_height = image.size
        width_ratio, height_ratio = width / image_width, height / image_height
        horizontal_image_width = int(image_width * width_ratio)
        horizontal_image_height = int(image_height * width_ratio)
        vertical_image_width = int(image_width * height_ratio) 
        vertical_image_height = int(image_height * height_ratio)
        top, left = 0, 0
        direction = None
        if width >= horizontal_image_width and height >= horizontal_image_height:
            input_image = image.resize((horizontal_image_width, horizontal_image_height))
            if anchor is not None:
                top_part, _ = anchor.split("-")
                if top_part == "center":
                    top = height // 2 - horizontal_image_height // 2
                elif top_part == "bottom":
                    top = height - horizontal_image_height
        elif width >= vertical_image_width and height >= vertical_image_height:
            input_image = image.resize((vertical_image_width, vertical_image_height))
            if anchor is not None:
                _, left_part = anchor.split("-")
                if left_part == "center":
                    left = width // 2 - vertical_image_width // 2
                elif left_part == "right":
                    left = width - vertical_image_width

        if offset_top is not None:
            top += offset_top
        if offset_left is not None:
            left += offset_left

        blank_image = Image.new("RGBA", (width, height))
        if input_image.mode == "RGBA":
            blank_image.paste(input_image, (left, top), input_image)
        else:
            blank_image.paste(input_image, (left, top))

        return blank_image

    elif fit == "cover":
        image_width, image_height = image.size
        width_ratio, height_ratio = width / image_width, height / image_height
        horizontal_image_width = math.ceil(image_width * width_ratio)
        horizontal_image_height = math.ceil(image_height * width_ratio)
        vertical_image_width = math.ceil(image_width * height_ratio)
        vertical_image_height = math.ceil(image_height * height_ratio)
        top, left = 0, 0
        direction = None
        if width <= horizontal_image_width and height <= horizontal_image_height:
            input_image = image.resize((horizontal_image_width, horizontal_image_height))
            if anchor is not None:
                top_part, _ = anchor.split("-")
                if top_part == "center":
                    top = height // 2 - horizontal_image_height // 2
                elif top_part == "bottom":
                    top = height - horizontal_image_height
        elif width <= vertical_image_width and height <= vertical_image_height:
            input_image = image.resize((vertical_image_width, vertical_image_height))
            if anchor is not None:
                _, left_part = anchor.split("-")
                if left_part == "center":
                    left = width // 2 - vertical_image_width // 2
                elif left_part == "right":
                    left = width - vertical_image_width
        else:
            input_image = image.resize((width, height))  # We're probably off by a pixel

        if offset_top is not None:
            top += offset_top
        if offset_left is not None:
            left += offset_left

        blank_image = Image.new("RGBA", (width, height))
        if input_image.mode == "RGBA":
            blank_image.paste(input_image, (left, top), input_image)
        else:
            blank_image.paste(input_image, (left, top))

        return blank_image

    elif fit == "stretch":
        return image.resize((width, height)).convert("RGBA")

    else:
        raise ValueError(f"Unknown fit {fit}")

def dilate_erode(
    image: Union[Image, List[Image]],
    value: int
) -> Union[Image, List[Image]]:
    """
    Given an image, dilate or erode it.
    Values of >0 dilate, <0 erode. 0 Does nothing.
    :see: http://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
    """
    if value == 0:
        return image
    if isinstance(image, list):
        return [
                dilate_erode(img, value) # type: ignore[misc]
            for img in image
        ]

    from PIL import Image
    import cv2 # type: ignore[import-not-found,unused-ignore]
    import numpy as np

    arr = np.array(image.convert("L"))
    transform = cv2.dilate if value > 0 else cv2.erode
    value = abs(value)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    arr = transform(arr, kernel, iterations=1)
    return Image.fromarray(arr)

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

def image_from_uri(uri: str) -> Image:
    """
    Loads an image using the fruition retriever if it's installed.
    That supports: http, file, ftp, ftps, sftp, and s3.
    If fruition is not installed, only supports local files and http.
    """
    from .download_util import retrieve_uri
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 2**32
    return Image.open(retrieve_uri(uri))

def images_are_equal(image_1: Image, image_2: Image) -> bool:
    """
    Determines if two images are equal.
    """
    from PIL import ImageChops
    if image_1.height != image_2.height or image_1.width != image_2.width:
        return False
    if image_1.mode == image_2.mode == "RGBA":
        image_1_alpha = [p[3] for p in image_1.getdata()] # type: ignore[attr-defined, unused-ignore]
        image_2_alpha = [p[3] for p in image_2.getdata()] # type: ignore[attr-defined, unused-ignore]
        if image_1_alpha != image_2_alpha:
            return False
    return not ImageChops.difference(
        image_1.convert("RGB"), image_2.convert("RGB")
    ).getbbox()

def image_pixelize(image: Image, factor: int = 2, exact: bool = True) -> Image:
    """
    Makes an image pixelized by downsizing and upsizing by a factor.
    """
    from PIL import Image
    from PIL.Image import Resampling
    width, height = image.size
    downsample_width = width // 2 ** factor
    downsample_height = height // 2 ** factor
    upsample_width = downsample_width * 2 ** factor if exact else width
    upsample_height = downsample_height * 2 ** factor if exact else height
    image = image.resize((downsample_width, downsample_height), resample=Resampling.NEAREST)
    image = image.resize((upsample_width, upsample_height), resample=Resampling.NEAREST)
    return image

def get_frames_or_image(image: Union[Image, List[Image]]) -> Union[Image, List[Image]]:
    """
    Makes sure an image is a list of images if it has more than one frame
    """
    if not isinstance(image, list):
        if getattr(image, "n_frames", 1) > 1:
            def get_frame(i: int) -> Image:
                image.seek(i)
                return image.copy().convert("RGB")
            return [
                get_frame(i)
                for i in range(image.n_frames) # type: ignore[attr-defined,unused-ignore]
            ]
    return image

def save_frames_or_image(
    image: Union[Image, List[Image]],
    directory: str,
    name: Optional[str]=None,
    video_format: str="webp",
    image_format: str="png"
) -> str:
    """
    Saves frames to image or video 
    """
    image = get_frames_or_image(image)
    if name is None:
        name = get_uuid()
    if isinstance(image, list):
        from .video_util import Video
        path = os.path.join(directory, f"{name}.{video_format}")
        Video(image).save(path)
    else:
        path = os.path.join(directory, f"{name}.{image_format}")
        image.save(path)
    return path

def get_frames_or_image_from_file(path: str) -> Union[Image, List[Image]]:
    """
    Opens a file to a single image or multiple
    """
    if path.startswith("data:"):
        # Should be a video
        if not path.startswith("data:video"):
            raise IOError(f"Received non-video data in video handler: {path}")
        # Dump to tempfile
        from tempfile import mktemp
        from base64 import b64decode
        header, _, data = path.partition(",")
        fmt, _, encoding = header.partition(";")
        _, _, file_ext = fmt.partition("/")
        dump_file = mktemp(f".{file_ext}")
        try:
            with open(dump_file, "wb") as fh:
                fh.write(b64decode(data))
            from .video_util import Video
            return Video.from_file(dump_file).frames_as_list
        finally:
            try:
                os.unlink(dump_file)
            except:
                pass
    else:
        name, ext = os.path.splitext(path)
        if ext in [".webp", ".webm", ".mp4", ".avi", ".mov", ".gif", ".m4v", ".mkv", ".ogg"]:
            from .video_util import Video
            return Video.from_file(path).frames_as_list
        else:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = 2**32
            image = Image.open(path)
            return get_frames_or_image(image)

def create_mask(
    width: int,
    height: int,
    left: int,
    top: int,
    right: int,
    bottom: int
) -> Image:
    """
    Creates a mask from 6 dimensions
    """
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    draw.rectangle(((left, top), (right, bottom)), fill="#ffffff")
    return image

def scale_image(
    image: Image,
    scale: Union[int, float]=1.0,
    nearest: int=1,
    smallest: Optional[int]=None,
) -> Image:
    """
    Scales an image proportionally.
    """
    from PIL import Image
    image_width, image_height = image.size
    width = float(image_width*scale)
    height = float(image_height*scale)
    if smallest is not None:
       smallest_side = min(width, height)
       if smallest_side < smallest:
            additional_upscale = smallest / smallest_side
            width *= additional_upscale
            height *= additional_upscale
    image_width = nearest * round(width / nearest)
    image_height = nearest * round(height / nearest)
    return image.resize((image_width, image_height), Image.Resampling.LANCZOS)

def image_tiles(
    image: Image,
    tile_size: Union[int, Tuple[int, int]],
    tile_stride: Union[int, Tuple[int, int]],
) -> Iterator[Image]:
    """
    Gets image tiles using sliding windows.
    """
    from .misc_util import sliding_windows
    width, height = image.size
    for x0, x1, y0, y1 in sliding_windows(width, height, tile_size, tile_stride):
        cropped = image.crop((x0, y0, x1, y1))
        setattr(cropped, "coordinates", (x0, y0, x1, y1))
        yield cropped

def get_image_metadata(image: Union[str, Image, List[Image]]) -> Dict[str, Any]:
    """
    Gets metadata from an image
    """
    if isinstance(image, str):
        return get_image_metadata(get_frames_or_image_from_file(image))
    elif isinstance(image, list):
        (width, height) = image[0].size
        return {
            "width": width,
            "height": height,
            "frames": len(image),
            "metadata": getattr(image[0], "text", {}),
        }
    else:
        (width, height) = image.size
        return {
            "width": width,
            "height": height,
            "metadata": getattr(image, "text", {})
        }

def redact_images_from_metadata(metadata: Dict[str, Any]) -> None:
    """
    Removes images from a metadata dictionary
    """
    for key in ["image", "mask"]:
        image = metadata.get(key, None)
        if image is not None:
            if isinstance(image, dict):
                image["image"] = get_image_metadata(image["image"])
            elif isinstance(image, str):
                metadata[key] = get_image_metadata(metadata[key])
            else:
                metadata[key] = get_image_metadata(metadata[key])
    if "control_images" in metadata:
        for i, control_dict in enumerate(metadata["control_images"]):
            control_dict["image"] = get_image_metadata(control_dict["image"])
    if "ip_adapter_images" in metadata:
        for i, ip_adapter_dict in enumerate(metadata["ip_adapter_images"]):
            ip_adapter_dict["image"] = get_image_metadata(ip_adapter_dict["image"])
    if "layers" in metadata:
        for layer in metadata["layers"]:
            redact_images_from_metadata(layer)

def image_to_tensor(
    image: Image,
    dtype: Optional[TorchDType]=None,
    mean: Optional[Union[List[float], Tuple[float, ...]]]=None,
    std: Optional[Union[List[float], Tuple[float, ...]]]=None,
) -> Tensor:
    """
    Converts a PIL image to a tensor
    """
    import torch
    import numpy as np
    torch_dtype = dtype if dtype is not None else torch.float
    img_array = np.array(image)
    if img_array.ndim == 2:  # Grayscale image
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = img_array.astype(np.float32) / 255.0
    if mean is not None and std is not None:
        img_array = (img_array - mean) / std

    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(torch_dtype)
    return img_tensor

def tensor_to_image(
    tensor: Tensor,
    mean: Optional[Union[List[float], Tuple[float, ...]]]=None,
    std: Optional[Union[List[float], Tuple[float, ...]]]=None,
) -> Image:
    """
    Converts a tensor to a PIL image
    """
    import torch
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 2**32
    if mean is not None and std is not None:
        mean_tensor = torch.tensor(mean).view(-1, 1, 1)
        std_tensor = torch.tensor(std).view(-1, 1, 1)
        tensor = tensor * std_tensor + mean_tensor
    tensor = (tensor * 255).byte()  # Denormalize to [0, 255]
    if tensor.ndimension() == 3:
        tensor = tensor.permute(1, 2, 0)  # Convert to HWC format
    if tensor.shape[-1] == 1:
        # Grayscale image, convert to RGB
        tensor = tensor.repeat(1, 1, 3)
    img_array = tensor.cpu().numpy()
    img = Image.fromarray(img_array)
    return img

def is_multi_image(images: ImageType) -> bool:
    """
    Determines if an image was intended to be a multi-frame image
    This is from the perspective of the developer, so generally this
    should check if the image is:
        1. A list of images
        2. A 4-dimensional tensor
        3. A 4-dimensional numpy array
    Others will be treated as singular images, so functions should return
    a singular image as well.
    """
    type_names = [
        mro_type.__name__
        for mro_type in type(images).mro()
    ]
    if "ndarray" in type_names:
        return len(images.shape) == 4 # type: ignore[union-attr]
    elif "Tensor" in type_names:
        return images.ndimension() == 4 # type: ignore[union-attr]
    return isinstance(images, list)

def get_image_width_height(image: ImageType) -> Tuple[int, int]:
    """
    Gets the size of an image
    """
    if isinstance(image, list):
        return get_image_width_height(image[0])

    type_names = [
        mro_type.__name__
        for mro_type in type(image).mro()
    ]
    if "Image" in type_names:
        return image.size # type: ignore
    elif "ndarray" in type_names:
        height, width = image.shape[:2] # type: ignore
        return width, height
    elif "Tensor" in type_names:
        height, width = image.shape[-2:] # type: ignore
        return width, height
    raise ValueError(f"Unsupported image type: {', '.join(type_names)}")

def assert_ndarray_image_num_channels(
    image: np.ndarray[Any, Any],
    num_channels: int
) -> np.ndarray[Any, Any]:
    """
    Converts between 1 (greyscale), 3 (RGB/RBG) and 4 (RGBA/RBGA) channel images
    """
    h, w, c = image.shape
    if c == num_channels:
        return image
    if num_channels not in [1, 3, 4]:
        raise ValueError(f"Unsupported number of target channels: {num_channels}")
    elif c not in [1, 3, 4]:
        raise ValueError(f"Unsupported number of image channels: {c}")

    if c == 1:
        if num_channels == 3:
            return np.repeat(image, 3, axis=-1)
        else:
            return np.concatenate([
                np.repeat(image, 3, axis=-1),
                np.ones((h, w, 1), dtype=image.dtype)
            ], axis=-1)
    elif c == 3:
        if num_channels == 1:
            return np.mean(image, axis=-1, keepdims=True) # type: ignore[no-any-return]
        else:
            return np.concatenate([
                image,
                np.ones((h, w, 1), dtype=image.dtype)
            ], axis=-1)
    else:
        if num_channels == 1:
            return np.mean(image[:, :, :3], axis=-1, keepdims=True) # type: ignore[no-any-return]
        else:
            return image[:, :, :4]

def to_bhwc_ndarray(
    images: ImageType,
    resize: Optional[Tuple[int, int]]=None,
    directory: Optional[str]=None,
    num_channels: Optional[int]=None,
) -> np.ndarray[Any, Any]:
    """
    Standardizes images to a 4D numpy array (BxHxWxC).

    Images can be:
    - A single PIL image
    - A list of PIL images
    - A single image or video URI
    - A list of image video URIs
    - A 3D or 4D numpy array (HxWxC or BxHxWxC) in the range [0,255], H/W/C may be swapped
    - A 3D or 4D torch tensor (CxHxW or BxCxHxW) in the range [0,1], C/H/W may be swapped
    >>> import numpy as np
    >>> base_image = np.zeros((100, 100, 3), dtype=np.uint8)
    >>> test_image_conversion = lambda im: np.allclose(to_bhwc_ndarray(im)[0], base_image)
    >>> assert test_image_conversion(base_image)
    >>> assert test_image_conversion(base_image[np.newaxis])
    >>> from PIL import Image
    >>> assert test_image_conversion(Image.new("RGB", (100, 100)))
    >>> assert test_image_conversion([Image.new("RGB", (100, 100))])
    >>> to_bhwc_ndarray(base_image, resize=(50, 50)).shape
    (1, 50, 50, 3)
    """
    import numpy as np

    if isinstance(images, (list, tuple)):
        return np.stack([
            bhwc_array
            for image in images
            for bhwc_array in to_bhwc_ndarray(
                image,
                resize=resize,
                directory=directory,
                num_channels=num_channels,
            )
        ])
    elif isinstance(images, str):
        if not os.path.isabs(images) and directory is not None:
            images = os.path.join(directory, images)

        image_list = get_frames_or_image_from_file(images)

        if isinstance(image_list, list):
            array_list = [
                np.array(img)
                for img in image_list
            ]
        else:
            array_list = [
                np.array(image_list)
            ]
    elif is_pil_image(images):
        array_list = [
            np.array(images)
        ]
    elif is_numpy_array(images):
        from PIL import Image
        if len(images.shape) == 3:
            array_list = [images]
        else:
            array_list = [
                i for i in images
            ]
    elif is_torch_tensor(images):
        if images.ndimension() == 3:
            array_list = [np.array(images.cpu().permute(1, 2, 0))]
        elif images.ndimension() == 4:
            array_list = [
                np.array(i.permute(1, 2, 0))
                for i in images.cpu()
            ]
        else:
            raise ValueError(f"Unsupported tensor shape: {images.shape}")
    else:
        raise ValueError(f"Unsupported image type: {type(images).__name__}")

    if resize is not None:
        # Use PIL
        from PIL import Image
        array_list = [
            np.array(i.resize(resize))
            for i in array_list
        ]
    if num_channels is not None:
        array_list = [
            assert_ndarray_image_num_channels(i, num_channels)
            for i in array_list
        ]

    return np.stack(array_list)

def assert_pil_num_channels(
    image: Image,
    num_channels: int
) -> Image:
    """
    Asserts that a PIL image has the correct number of channels
    """
    if num_channels == 1:
        return image.convert("L")
    elif num_channels == 3:
        return image.convert("RGB")
    elif num_channels == 4:
        return image.convert("RGBA")
    raise ValueError(f"Unsupported number of channels: {num_channels}")

def to_pil_array(
    images: ImageType,
    resize: Optional[Tuple[int, int]]=None,
    mean: Optional[Union[List[float], Tuple[float, ...]]]=None,
    std: Optional[Union[List[float], Tuple[float, ...]]]=None,
    directory: Optional[str]=None,
    num_channels: Optional[int]=None,
) -> List[Image]:
    """
    Standardizes images to a list of PIL images.

    Images can be:
    - A single PIL image
    - A list of PIL images
    - A single image or video URI
    - A list of image video URIs
    - A 3D or 4D numpy array (HxWxC or BxHxWxC) in the range [0,255], H/W/C may be swapped
    - A 3D or 4D torch tensor (CxHxW or BxCxHxW) in the range [0,1], C/H/W may be swapped
    >>> from PIL import Image
    >>> base_image = Image.new("RGB", (100, 100))
    >>> test_image_conversion = lambda im: images_are_equal(to_pil_array(im)[0], base_image)
    >>> assert test_image_conversion(base_image)
    >>> assert test_image_conversion([base_image])
    >>> import numpy as np
    >>> assert test_image_conversion(np.zeros((100, 100, 3), dtype=np.uint8))
    >>> assert test_image_conversion(np.zeros((1, 100, 100, 3), dtype=np.uint8))
    >>> import torch
    >>> assert test_image_conversion(torch.zeros(3, 100, 100, dtype=torch.float))
    >>> assert test_image_conversion(torch.zeros(1, 3, 100, 100, dtype=torch.half))
    >>> to_pil_array(base_image, resize=(50, 50))[0].size
    (50, 50)
    """
    if isinstance(images, (list, tuple)):
        return [
            pil_image
            for image in images
            for pil_image in to_pil_array(
                image,
                resize=resize,
                mean=mean,
                std=std,
                directory=directory,
                num_channels=num_channels,
            )
        ]
    elif isinstance(images, str):
        if not os.path.isabs(images) and directory is not None:
            images = os.path.join(directory, images)

        image_list = get_frames_or_image_from_file(images)

        if isinstance(image_list, list):
            pil_images = image_list
        else:
            pil_images = [image_list]
    elif is_pil_image(images):
        pil_images = [images]
    elif is_numpy_array(images):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 2**32

        if len(images.shape) == 3:
            pil_images = [
                Image.fromarray(images)
            ]
        else:
            pil_images = [
                Image.fromarray(i)
                for i in images
            ]
    elif is_torch_tensor(images):
        if images.ndimension() == 3:
            pil_images = [
                tensor_to_image(
                    images,
                    mean=mean,
                    std=std
                )
            ]
        elif images.ndimension() == 4:
            pil_images = [
                tensor_to_image(
                    i,
                    mean=mean,
                    std=std
                )
                for i in images
            ]
        else:
            raise ValueError(f"Unsupported tensor shape: {images.shape}")
    else:
        raise ValueError(f"Unsupported image type: {type(images).__name__}")

    if resize is not None:
        pil_images = [
            i.resize(resize)
            for i in pil_images
        ]
    if num_channels is not None:
        pil_images = [
            assert_pil_num_channels(i, num_channels)
            for i in pil_images
        ]

    return pil_images

def assert_torch_image_num_channels(
    image: Tensor,
    num_channels: int
) -> Tensor:
    """
    Converts between 1 (greyscale), 3 (RGB/RBG) and 4 (RGBA/RBGA) channel images
    """
    c, h, w = image.shape
    if c == num_channels:
        return image

    if num_channels not in [1, 3, 4]:
        raise ValueError(f"Unsupported number of target channels: {num_channels}")
    elif c not in [1, 3, 4]:
        raise ValueError(f"Unsupported number of image channels: {c}")

    import torch
    if c == 1:
        if num_channels == 3:
            return image.repeat(3, 1, 1)
        else:
            return torch.cat([
                image.repeat(3, 1, 1),
                torch.ones(1, h, w, dtype=image.dtype)
            ], dim=0)
    elif c == 3:
        if num_channels == 1:
            return image.mean(dim=0, keepdim=True)
        else:
            return torch.cat([
                image,
                torch.ones(1, h, w, dtype=image.dtype)
            ], dim=0)
    else:
        if num_channels == 1:
            return image[:3].mean(dim=0, keepdim=True)
        else:
            return image[:3]

def to_bchw_tensor(
    images: ImageType,
    resize: Optional[Tuple[int, int]]=None,
    dtype: Optional[TorchDType]=None,
    mean: Optional[Union[List[float], Tuple[float, ...]]]=None,
    std: Optional[Union[List[float], Tuple[float, ...]]]=None,
    num_channels: Optional[int]=None,
    directory: Optional[str]=None,
) -> Tensor:
    """
    Standardizes images to a 4D tensor (BxCxHxW) in the range [0, 1].

    Images can be:
    - A single PIL image
    - A list of PIL images
    - A single image or video URI
    - A list of image or video URIs
    - A 3D or 4D numpy array (HxWxC or BxHxWxC), H/W/C may be swapped
    - A 3D or 4D torch tensor (CxHxW or BxCxHxW), C/H/W may be swapped

    >>> import torch
    >>> base_image = torch.zeros(3, 100, 100, dtype=torch.float)
    >>> test_image_conversion = lambda im: torch.allclose(to_bchw_tensor(im)[0], base_image)
    >>> assert test_image_conversion(base_image)
    >>> assert test_image_conversion(base_image.unsqueeze(0))
    >>> import numpy as np
    >>> assert test_image_conversion(np.zeros((100, 100, 3), dtype=np.uint8))
    >>> assert test_image_conversion(np.zeros((1, 100, 100, 3), dtype=np.uint8))
    >>> from PIL import Image
    >>> assert test_image_conversion(Image.new("RGB", (100, 100)))
    >>> assert test_image_conversion([Image.new("RGB", (100, 100))])
    >>> to_bchw_tensor(base_image, resize=(50, 50)).shape
    torch.Size([1, 3, 50, 50])
    """
    import torch

    if isinstance(images, (list, tuple)):
        return torch.stack([
            bchw_tensor
            for image in images
            for bchw_tensor in to_bchw_tensor(
                image,
                resize=resize,
                dtype=dtype,
                mean=mean,
                std=std,
                num_channels=num_channels,
                directory=directory,
            )
        ])
    elif isinstance(images, str):
        if not os.path.isabs(images) and directory is not None:
            images = os.path.join(directory, images)
        images = get_frames_or_image_from_file(images)
        return to_bchw_tensor(
            images,
            resize=resize,
            dtype=dtype,
            mean=mean,
            std=std,
            num_channels=num_channels,
            directory=directory,
        )

    if dtype is not None:
        from .torch_util import get_torch_dtype
        torch_dtype = get_torch_dtype(dtype)
    else:
        torch_dtype = None

    if is_pil_image(images):
        maybe_image_list = get_frames_or_image(images)
        if isinstance(maybe_image_list, list):
            tensor_list = [
                image_to_tensor(i, dtype=torch_dtype, mean=mean, std=std)
                for i in maybe_image_list
            ]
        else:
            tensor_list = [
                image_to_tensor(maybe_image_list, dtype=torch_dtype, mean=mean, std=std)
            ]
    elif is_numpy_array(images):
        if images.ndim == 3:
            tensor_list = [
                torch.tensor(images, dtype=torch_dtype).permute(2, 0, 1) / 255.0
            ]
        elif images.ndim == 4:
            tensor_list = [
                torch.tensor(i, dtype=torch_dtype).permute(2, 0, 1) / 255.0
                for i in images
            ]
        else:
            raise ValueError(f"Unsupported numpy array shape: {images.shape}")
    elif is_torch_tensor(images):
        if resize is not None:
            from .torch_util import scale_tensor
            images = scale_tensor(images, size=resize)
        if images.ndimension() == 3:
            tensor_list = [images.to(dtype=torch_dtype)]
        elif images.ndimension() == 4:
            tensor_list = [
                i
                for i in images.to(dtype=torch_dtype)
            ]
        else:
            raise ValueError(f"Unsupported tensor shape: {images.shape}")
    else:
        raise ValueError(f"Unsupported image type: {type(images).__name__}")

    # Make sure values are in range 0, 1
    min_value = min(i.min().item() for i in tensor_list)
    max_value = max(i.max().item() for i in tensor_list)

    if min_value < -1:
        # Scale linearly using the range
        value_range = max_value - min_value
        tensor_list = [
            (i - min_value) / value_range
            for i in tensor_list
        ]
    elif min_value < 0:
        if max_value > 1:
            # Turn [-1, n] to [-1, 1]
            # Scale linearly using the range
            tensor_list = [
                i / max_value
                for i in tensor_list
            ]
        # Turn [-1, 1] to [0, 1]
        tensor_list = [
            (i + 1) / 2
            for i in tensor_list
        ]
    elif max_value > 255:
        raise ValueError(f"Maximum value is greater than 255: {max_value}")
    elif max_value > 1:
        # Turn [0, 255] to [0, 1]
        tensor_list = [
            i / 255.0
            for i in tensor_list
        ]

    if resize is not None:
        from .torch_util import scale_tensor
        tensor_list = [
            scale_tensor(i, size=resize)
            for i in tensor_list
        ]

    if num_channels is not None:
        tensor_list = [
            assert_torch_image_num_channels(i, num_channels)
            for i in tensor_list
        ]

    if mean is not None and std is not None:
        mean_tensor = torch.tensor(mean).view(-1, 1, 1)
        std_tensor = torch.tensor(std).view(-1, 1, 1)
        tensor_list = [
            i - mean_tensor / std_tensor
            for i in tensor_list
        ]
    
    return torch.stack(tensor_list)

_is_turbojpeg_available: Optional[bool] = None
def is_turbojpeg_available() -> bool:
    """
    Determines if TurboJPEG is available
    """
    global _is_turbojpeg_available
    if _is_turbojpeg_available is None:
        try:
            import turbojpeg # type: ignore[import-not-found,import-untyped,unused-ignore]
            _is_turbojpeg_available = True
        except ImportError:
            _is_turbojpeg_available = False
    return _is_turbojpeg_available

_turbojpeg_encoder: Optional[TurboJPEG] = None
def get_turbojpeg_encoder() -> TurboJPEG:
    """
    Gets a TurboJPEG encoder
    """
    global _turbojpeg_encoder
    if _turbojpeg_encoder is None:
        from turbojpeg import TurboJPEG # type: ignore[import-not-found,import-untyped,unused-ignore]
        _turbojpeg_encoder = TurboJPEG()
    return _turbojpeg_encoder

def encode_jpeg(image_data: np.ndarray[Any, Any], quality: int=95) -> bytes:
    """
    Encodes a JPEG either using PIL or TurboJPEG
    """
    import numpy as np
    if np.issubdtype(image_data.dtype, np.floating):
        image_data = (image_data * 255).astype(np.uint8)
    if is_turbojpeg_available():
        return get_turbojpeg_encoder().encode(image_data, quality=quality, pixel_format=0) # type: ignore[no-any-return]
    from PIL import Image
    with io.BytesIO() as output:
        Image.fromarray(image_data).save(output, format="JPEG", quality=quality)
        return output.getvalue()

def to_jpeg_array(
    images: ImageType,
    resize: Optional[Tuple[int, int]]=None,
    directory: Optional[str]=None,
) -> List[EncodedImageProxy]:
    """
    Standardizes images to a list of JPEG-encoded images.
    Uses the TJPEG format for encoding.
    """
    # First standardize
    images = to_bhwc_ndarray(images, resize=resize, directory=directory)
    # Then encode
    return [
        EncodedImageProxy(
            data=encode_jpeg(image),
            format="jpeg",
            size=(image.shape[1], image.shape[0])
        )
        for image in images
    ]

def serialize_image(image: Image) -> str:
    """
    Serializes an image to a base64 string
    """
    from PIL.PngImagePlugin import PngInfo
    with io.BytesIO() as output:
        info = PngInfo()
        text_metadata = getattr(image, "text", {})
        for key in text_metadata:
            info.add_text(key, text_metadata[key])
        image.save(output, format="PNG", pnginfo=info)
        image_bytestring = base64.b64encode(output.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_bytestring}"

def pad_image(
    image: ImageType, 
    padding: Union[int, Tuple[int, int, int, int]],
) -> ImageType:
    """
    Pads an image

    :param image: The image to pad. Either a PIL image, numpy array, or torch tensor.
    :param padding: The padding to apply. Can be an integer or a tuple of 4 integers.
    :return: The padded image.
    """
    from PIL import ImageOps
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)

    if is_pil_image(image):
        return ImageOps.expand(image, padding)
    elif is_numpy_array(image):
        import numpy as np
        return np.pad(image, ((padding[0], padding[1]), (padding[2], padding[3]), (0, 0)))
    elif is_torch_tensor(image):
        import torch
        return torch.nn.functional.pad(image, padding)
    else:
        raise ValueError(f"Unsupported image type: {type(image).__name__}")

def pad_image_to_nearest(
    image: ImageType,
    multiple: int,
    return_padding: bool = False,
    return_crop: bool = False,
) -> Union[
    ImageType,
    Tuple[
        ImageType, Tuple[int, int, int, int]
    ]
]:
    """
    Pads an image to the nearest multiple

    :param image: The image to pad. Either a PIL image, numpy array, or torch tensor.
    :param multiple: The multiple to pad to.
    :return: The padded image.
    """
    if is_pil_image(image):
        width, height = image.size
    elif is_numpy_array(image):
        height, width = image.shape[:2]
    elif is_torch_tensor(image):
        height, width = image.shape[-2:]
    else:
        raise ValueError(f"Unsupported image type: {type(image).__name__}")

    if width % multiple == 0:
        width_padding = 0
    else:
        width_padding = (multiple - width % multiple) % multiple
    if height % multiple == 0:
        height_padding = 0
    else:
        height_padding = (multiple - height % multiple) % multiple

    left_padding = width_padding // 2
    right_padding = width_padding - left_padding
    top_padding = height_padding // 2
    bottom_padding = height_padding - top_padding

    padding = (top_padding, bottom_padding, left_padding, right_padding)
    padded_image = pad_image(image, padding)

    if return_padding:
        return padded_image, padding
    elif return_crop:
        return padded_image, (top_padding, left_padding, height + height_padding, width + width_padding)
    return padded_image

def show_image(
    image: ImageType,
    title: Optional[str]=None,
) -> None:
    """
    Shows an image
    """
    image = to_pil_array(image)[0]
    try:
        from IPython.display import display # type: ignore[import-not-found]
        display(image)
    except ImportError:
        import cv2
        import numpy as np
        # Show until a key is pressed
        cv2.imshow(title or "Image", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
