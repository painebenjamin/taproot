from __future__ import annotations
# Adapted from https://raw.githubusercontent.com/lhwcv/mlsd_pytorch/main/utils.py
# modified by lihaoweicv
# modified again by benjamin paine
# M-LSD
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

from typing import Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import numpy as np
    from PIL import Image

__all__ = [
    "decode_output_score_and_ptss",
    "pred_lines",
    "draw_lines",
]

def decode_output_score_and_ptss(
    output: torch.Tensor,
    top_k: int=200,
    k_size: int=5
) -> Tuple[np.ndarray[Any, Any], ...]:
    """
    Decode output into components
    """
    import torch
    import numpy as np
    from torch.nn import functional as F
    b, c, h, w = output.shape
    assert b == 1, "only support bsize==1"

    center = output[:, 0, :, :]
    displacement = output[:, 1:5, :, :][0]

    heat = torch.sigmoid(center)
    hmax = F.max_pool2d(
        heat,
        (k_size, k_size),
        stride=1,
        padding=(k_size - 1) // 2
    )
    keep = (hmax == heat).float()
    heat = heat * keep
    heat = heat.reshape(-1)

    scores, indices = torch.topk(heat, top_k, dim=-1, largest=True)
    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    ptss = torch.cat((yy, xx), dim=-1)

    if torch.isnan(scores).any():
        raise ValueError("NaN in scores")
    if torch.isnan(ptss).any():
        raise ValueError("NaN in ptss")
    if torch.isnan(displacement).any():
        raise ValueError("NaN in displacement")

    ptss = ptss.detach().cpu()
    scores = scores.detach().cpu()
    displacement = displacement.detach().cpu()

    return tuple([x.numpy() for x in (ptss, scores, displacement)])

def pred_lines(
    outputs: torch.Tensor,
    score_thr: float=0.10,
    dist_thr: float=20.0,
) -> np.ndarray[Any, Any]:
    """
    Predict lines from outputs
    """
    import numpy as np
    pts, pts_score, vmap = decode_output_score_and_ptss(outputs, 200, 3)

    vmap = vmap.transpose((1, 2, 0))

    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]

    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    if not segments_list:
        return np.zeros((0, 4))

    lines = 2 * np.array(segments_list)  # 256 > 512
    return lines

def draw_lines(
    width: int,
    height: int,
    lines: np.ndarray[Any, Any]
) -> Image.Image:
    """
    Draw lines on image
    """
    from PIL import Image
    import numpy as np
    import cv2 # type: ignore[import-not-found]

    img = np.zeros((height, width, 3), np.uint8)
    w_ratio = width / 512
    h_ratio = height / 512

    for line in lines:
        cv2.line(
            img,
            (int(line[0] * w_ratio), int(line[1] * h_ratio)),
            (int(line[2] * w_ratio), int(line[3] * h_ratio)),
            (255, 255, 255),
            1,
            16,
        )

    return Image.fromarray(img).resize((width, height))
