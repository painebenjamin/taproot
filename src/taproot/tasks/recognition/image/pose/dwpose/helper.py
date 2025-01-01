# type: ignore
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by Enfugue
# 6th Edited by Taproot

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np

from PIL import Image

from taproot.util import hwc3, safe_resize

from .util import (
    draw_bodypose,
    draw_handpose,
    draw_facepose,
)

# override mmengine module registry to ignore keyerrors
from mmengine.registry.registry import Registry
original_register = Registry._register_module
def _register_module(self, module, module_name=None, force=False):
    try:
        original_register(self, module, module_name, force)
    except KeyError:
        pass
Registry._register_module = _register_module

def draw_pose(pose, H, W, draw_type="pose"):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]

    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    if draw_type == "pose":
        canvas = draw_bodypose(canvas, candidate, subset)
    canvas = draw_handpose(canvas, hands, draw_type=draw_type)
    canvas = draw_facepose(canvas, faces, draw_type=draw_type)

    return canvas

class DWPoseDetector:
    def __init__(
        self,
        det_config=None,
        det_ckpt=None,
        pose_config=None,
        pose_ckpt=None,
        device="cpu",
    ):
        from .wholebody import Wholebody

        self.pose_estimation = Wholebody(
            det_config, det_ckpt, pose_config, pose_ckpt, device
        )

    def to(self, device, dtype=None):
        self.pose_estimation.to(device, dtype=dtype)
        return self

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        output_type="pil",
        draw_type="pose",
        isolated=False,
        **kwargs
    ):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image = hwc3(input_image)
        input_image = safe_resize(input_image, detect_resolution)
        H, W, C = input_image.shape
        output_img = safe_resize(input_image, image_resolution)
        oH, oW, oC = output_img.shape
        min_sum = 3 * 255 * 128

        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18]

            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            if isolated:
                all_maps = []
                for key in ["bodies", "hands", "faces"]:
                    if (key == "bodies" and pose[key]["subset"].any()) or (key != "bodies" and pose[key].any()):
                        these_maps = []
                        this_pose = {"bodies": {"candidate": pose["bodies"]["candidate"], "subset": []}, "hands": [], "faces": []}
                        if key == "bodies":
                            parts = pose[key]["subset"]
                        else:
                            parts = pose[key]
                        for j, part in enumerate(parts):
                            if key == "bodies":
                                this_pose[key]["subset"] = [part]
                            else:
                                this_pose[key] = [part]
                            detected_map = draw_pose(this_pose, H, W, draw_type)
                            if np.sum(detected_map) > min_sum:
                                these_maps.append(detected_map)
                        all_maps.extend(these_maps)
                for j, detected_map in enumerate(all_maps):
                    detected_map = hwc3(detected_map)
                    detected_map = cv2.resize(
                        detected_map, (oW, oH), interpolation=cv2.INTER_LINEAR
                    )
                    if output_type == "pil":
                        detected_map = Image.fromarray(detected_map)
                    all_maps[j] = detected_map

                return all_maps
            else:
                detected_map = draw_pose(pose, H, W, draw_type)
                detected_map = hwc3(detected_map)
                detected_map = cv2.resize(
                    detected_map, (oW, oH), interpolation=cv2.INTER_LINEAR
                )
                if output_type == "pil":
                    detected_map = Image.fromarray(detected_map)

                return detected_map
