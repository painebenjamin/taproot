# type: ignore
# Adapted from https://github.com/patrickvonplaten/controlnet_aux/blob/master/src/controlnet_aux/open_pose/__init__.py
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import List, NamedTuple, Tuple, Union

import cv2
import numpy as np
import torch

from PIL import Image

from taproot.util import (
    hwc3,
    safe_resize,
)
from .body import Body, BodyResult, Keypoint
from .face import Face
from .hand import Hand
from .util import (
    draw_bodypose,
    draw_handpose,
    draw_facepose,
    handDetect,
    faceDetect
)

HandResult = List[Keypoint]
FaceResult = List[Keypoint]

class PoseResult(NamedTuple):
    body: BodyResult
    left_hand: Union[HandResult, None]
    right_hand: Union[HandResult, None]
    face: Union[FaceResult, None]


def draw_poses(
    poses: List[PoseResult],
    H,
    W,
    draw_body=True,
    draw_hand=True,
    draw_face=True,
    draw_type="pose"
):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = draw_bodypose(canvas, pose.body.keypoints)

        if draw_hand:
            canvas = draw_handpose(canvas, pose.left_hand, draw_type=draw_type)
            canvas = draw_handpose(canvas, pose.right_hand, draw_type=draw_type)

        if draw_face:
            canvas = draw_facepose(canvas, pose.face, draw_type=draw_type)

    return canvas


class OpenposeDetector:
    """
    A class for detecting human poses in images using the Openpose model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """

    def __init__(self) -> None:
        self.body_estimation = Body()
        self.hand_estimation = Hand()
        self.face_estimation = Face()

    @property
    def modules(self) -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict({
            "body": self.body_estimation.model,
            "hand": self.hand_estimation.model,
            "face": self.face_estimation.model,
        })

    def to(self, device, dtype=None):
        self.body_estimation.to(device, dtype=dtype)
        self.hand_estimation.to(device, dtype=dtype)
        self.face_estimation.to(device, dtype=dtype)
        return self

    def detect_hands(
        self, body: BodyResult, oriImg
    ) -> Tuple[Union[HandResult, None], Union[HandResult, None]]:
        left_hand = None
        right_hand = None
        H, W, _ = oriImg.shape
        for x, y, w, is_left in handDetect(body, oriImg):
            peaks = self.hand_estimation(oriImg[y : y + w, x : x + w, :]).astype(
                np.float32
            )
            if peaks.ndim == 2 and peaks.shape[1] == 2:
                peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(
                    W
                )
                peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(
                    H
                )

                hand_result = [Keypoint(x=peak[0], y=peak[1]) for peak in peaks]

                if is_left:
                    left_hand = hand_result
                else:
                    right_hand = hand_result

        return left_hand, right_hand

    def detect_face(self, body: BodyResult, oriImg) -> Union[FaceResult, None]:
        face = faceDetect(body, oriImg)
        if face is None:
            return None

        x, y, w = face
        H, W, _ = oriImg.shape
        heatmaps = self.face_estimation(oriImg[y : y + w, x : x + w, :])
        peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(
            np.float32
        )
        if peaks.ndim == 2 and peaks.shape[1] == 2:
            peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
            peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
            return [Keypoint(x=peak[0], y=peak[1]) for peak in peaks]

        return None

    def detect_poses(
        self, oriImg, include_hand=False, include_face=False
    ) -> List[PoseResult]:
        """
        Detect poses in the given image.
            Args:
                oriImg (numpy.ndarray): The input image for pose detection.
                include_hand (bool, optional): Whether to include hand detection. Defaults to False.
                include_face (bool, optional): Whether to include face detection. Defaults to False.

        Returns:
            List[PoseResult]: A list of PoseResult objects containing the detected poses.
        """
        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            bodies = self.body_estimation.format_body_result(candidate, subset)

            results = []
            for body in bodies:
                left_hand, right_hand, face = (None,) * 3
                if include_hand:
                    left_hand, right_hand = self.detect_hands(body, oriImg)
                if include_face:
                    face = self.detect_face(body, oriImg)

                results.append(
                    PoseResult(
                        BodyResult(
                            keypoints=[
                                Keypoint(
                                    x=keypoint.x / float(W), y=keypoint.y / float(H)
                                )
                                if keypoint is not None
                                else None
                                for keypoint in body.keypoints
                            ],
                            total_score=body.total_score,
                            total_parts=body.total_parts,
                        ),
                        left_hand,
                        right_hand,
                        face,
                    )
                )

            return results

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        include_body=True,
        include_hand=False,
        include_face=False,
        hand_and_face=None,
        draw_type="pose",
        isolated=False,
        **kwargs
    ):
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = hwc3(input_image)
        input_image = safe_resize(input_image, detect_resolution)
        H, W, C = input_image.shape

        output_img = safe_resize(input_image, image_resolution)
        oH, oW, oC = output_img.shape

        poses = self.detect_poses(input_image, include_hand, include_face)

        min_sum = 3 * 255 * 12

        if isolated:
            all_masks = []
            for pose in poses:
                these_masks = []
                if include_body:
                    canvas = draw_poses(
                        [pose],
                        H,
                        W,
                        draw_body=True,
                        draw_hand=False,
                        draw_face=False,
                        draw_type=draw_type,
                    )
                    if np.sum(canvas) > min_sum:
                        these_masks.append(canvas)
                if include_hand:
                    canvas = draw_poses(
                        [pose],
                        H,
                        W,
                        draw_body=False,
                        draw_hand=True,
                        draw_face=False,
                        draw_type=draw_type,
                    )
                    if np.sum(canvas) > min_sum:
                        these_masks.append(canvas)
                if include_face:
                    canvas = draw_poses(
                        [pose],
                        H,
                        W,
                        draw_body=False,
                        draw_hand=False,
                        draw_face=True,
                        draw_type=draw_type,
                    )
                    if np.sum(canvas) > min_sum:
                        these_masks.append(canvas)
                all_masks.extend(these_masks)
            for i, detected_map in enumerate(all_masks):
                detected_map = hwc3(detected_map)
                detected_map = cv2.resize(detected_map, (oW, oH), interpolation=cv2.INTER_LINEAR)
                all_masks[i] = Image.fromarray(detected_map)
            return all_masks
        else:
            canvas = draw_poses(
                poses,
                H,
                W,
                draw_body=include_body,
                draw_hand=include_hand,
                draw_face=include_face,
                draw_type=draw_type,
            )

            detected_map = canvas
            detected_map = hwc3(detected_map)
            detected_map = cv2.resize(detected_map, (oW, oH), interpolation=cv2.INTER_LINEAR)
            detected_map = Image.fromarray(detected_map)

            return detected_map
