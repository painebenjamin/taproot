from __future__ import annotations
from taproot.util import (
    debug_logger,
    get_test_image,
    get_test_images,
    save_test_image,
    get_test_result,
    execute_task_test_suite,
)

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from PIL import Image

def test_openpose() -> None:
    """
    Test the openpose detection task.
    """
    image_size = "512x512"
    subject = "person"
    number = 2

    with debug_logger() as logger:
        # Base pose test
        person_image = get_test_image(
            subject=subject,
            size=image_size,
            number=number
        )

        def get_test_result_image(prefix: str) -> Optional[Image.Image]:
            try:
                return get_test_result(
                    subject=f"{prefix}_openpose",
                    size=image_size,
                    number=1
                )
            except FileNotFoundError:
                return None

        try:
            person_image_result = get_test_result(
                subject="openpose",
                size="512x512",
                number=1
            )
        except FileNotFoundError:
            person_image_result = None

        pose_result = get_test_result_image("pose")
        no_body_result = get_test_result_image("nobody_pose")
        no_face_result = get_test_result_image("noface_pose")
        no_hands_result = get_test_result_image("nohands_pose")
        face_mask_result = get_test_result_image("face_mask")
        hand_mask_result = get_test_result_image("hand_mask")

        if face_mask_result is None or hand_mask_result is None:
            non_composite_result = None
        else:
            non_composite_result = [
                hand_mask_result,
                face_mask_result,
            ]

        test_results = execute_task_test_suite(
            "pose-detection",
            model="openpose",
            cases=[
                ({"image": person_image}, pose_result),
                ({"image": person_image, "body": False}, no_body_result),
                ({"image": person_image, "face": False}, no_face_result),
                ({"image": person_image, "hands": False}, no_hands_result),
                ({"image": person_image, "mode": "mask", "composite": False}, non_composite_result),
            ],
        )

        if pose_result is None:
            save_test_image(
                test_results[0],
                "pose_openpose"
            )
        if no_body_result is None:
            save_test_image(
                test_results[1],
                "nobody_pose_openpose"
            )
        if no_face_result is None:
            save_test_image(
                test_results[2],
                "noface_pose_openpose"
            )
        if no_hands_result is None:
            save_test_image(
                test_results[3],
                "nohands_pose_openpose"
            )
        if hand_mask_result is None:
            save_test_image(
                test_results[4][0],
                "hand_mask_openpose"
            )
        if face_mask_result is None:
            save_test_image(
                test_results[4][1],
                "face_mask_openpose"
            )

def run_batch(num_images: int) -> None:
    """
    Test with a batch size.
    """
    with debug_logger() as logger:
        people_images = get_test_images(
            num_images=num_images,
            subject="person",
            size="512x512"
        )
        execute_task_test_suite(
            "pose-detection",
            model="openpose",
            cases=[
                ({"image": people_images}, None)
            ]
        )

def test_batch_5() -> None:
    """
    Test with a batch size of 5.
    """
    run_batch(5)
