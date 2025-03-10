# type: ignore
# Adapted from https://github.com/patrickvonplaten/controlnet_aux/master/src/controlnet_aux/pidi/__init__.py

import warnings
import cv2
import numpy as np
import torch

from einops import rearrange
from PIL import Image

from taproot.util import hwc3, nms_mask, safe_resize, safe_step

from .model import pidinet

class PidiNetDetector:
    def __init__(self):
        self.netNetwork = pidinet()

    def to(self, device, dtype=None):
        self.netNetwork.to(device, dtype=dtype)
        return self

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        safe=False,
        output_type="pil",
        scribble=False,
        apply_filter=False,
        **kwargs
    ):
        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
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
        input_image = input_image[:, :, ::-1].copy()
        with torch.no_grad():
            image_pidi = torch.from_numpy(input_image).to(device, dtype=dtype)
            image_pidi = image_pidi / 255.0
            image_pidi = rearrange(image_pidi, "h w c -> 1 c h w")
            edge = self.netNetwork(image_pidi)[-1]
            edge = edge.cpu().numpy()
            if apply_filter:
                edge = edge > 0.5
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge[0, 0]
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
