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

def test_dwpose() -> None:
    """
    Test the dwpose detection task.
    """
    image_size = "512x512"
    subject = "person"
    number = 8

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
                    subject=f"{prefix}_dwpose",
                    size=image_size,
                    number=1
                )
            except FileNotFoundError:
                return None

        pose_result = get_test_result_image("pose")
        face_mask_result = get_test_result_image("face_mask")
        hand_mask_result = get_test_result_image("hand_mask")

        test_results = execute_task_test_suite(
            "pose-detection",
            model="dwpose",
            cases=[
                ({"image": person_image}, pose_result),
                ({"image": person_image, "mode": "mask", "hands": False}, face_mask_result),
                ({"image": person_image, "mode": "mask", "face": False}, hand_mask_result),
            ],
        )

        if pose_result is None:
            save_test_image(
                test_results[0],
                "pose_dwpose"
            )
        if face_mask_result is None:
            save_test_image(
                test_results[1],
                "face_mask_dwpose"
            )
        if hand_mask_result is None:
            save_test_image(
                test_results[2],
                "hand_mask_dwpose"
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
            model="dwpose",
            cases=[
                ({"image": people_images}, None)
            ]
        )

def test_batch_5() -> None:
    """
    Test with a batch size of 5.
    """
    run_batch(5)
