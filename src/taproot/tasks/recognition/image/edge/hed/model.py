# type: ignore
from __future__ import annotations
# Adapted from https://raw.githubusercontent.com/patrickvonplaten/controlnet_aux/master/src/controlnet_aux/hed/__init__.py
# This is an improved version and model of HED edge detection with Apache License, Version 2.0.
# Please use this implementation in your products
# This implementation may produce slightly different results from Saining Xie's official implementations,
# but it generates smoother edges and is more suitable for ControlNet as well as other image-to-image translations.
# Different from official models and other implementations, this is an RGB-input model (rather than BGR)
# and in this way it works better for gradio's RGB protocol

import warnings
import cv2
import numpy as np
import torch

from typing import Tuple, Union, Optional

from PIL import Image
from einops import rearrange

from taproot.util import (
    hwc3,
    nms_mask,
    safe_resize,
    safe_step,
)

class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel: int, output_channel: int, layer_number: int) -> None:
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(
            torch.nn.Conv2d(
                in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1
            )
        )
        for i in range(1, layer_number):
            self.convs.append(
                torch.nn.Conv2d(
                    in_channels=output_channel,
                    out_channels=output_channel,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                )
            )
        self.projection = torch.nn.Conv2d(
            in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0
        )

    def __call__(self, x: torch.Tensor, down_sampling: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class HEDDetector:
    def __init__(self) -> None:
        self.netNetwork = ControlNetHED_Apache2()

    def to(self, device: torch.device, dtype: Optional[torch.dtype]=None) -> "HEDDetector":
        self.netNetwork.to(device, dtype=dtype)
        return self

    def __call__(
        self,
        input_image: Union[np.ndarray, Image.Image],
        detect_resolution: int = 512,
        image_resolution: int = 512,
        safe: bool = False,
        output_type: str = "pil",
        scribble: bool = False,
        **kwargs
    ) -> Union[np.ndarray, Image.Image]:
        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"

        if isinstance(output_type, bool):
            warnings.warn(
                "Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions"
            )
            if output_type:
                output_type = "pil"

        test_param = next(iter(self.netNetwork.parameters()))
        device = test_param.device
        dtype = test_param.dtype

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = hwc3(input_image)
        input_image = safe_resize(input_image, detect_resolution)

        assert input_image.ndim == 3
        H, W, C = input_image.shape
        with torch.no_grad():
            image_hed = torch.from_numpy(input_image.copy()).to(device, dtype=dtype)
            image_hed = rearrange(image_hed, "h w c -> 1 c h w")
            edges = self.netNetwork(image_hed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge
        detected_map = hwc3(detected_map)

        img = safe_resize(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if scribble:
            detected_map = nms_mask(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
