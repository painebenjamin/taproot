from taproot.util import (
    debug_logger,
    get_test_image,
    save_test_image,
    get_test_result,
    execute_task_test_suite,
)

def test_midas_depth() -> None:
    """
    Test the midas (depth detection) task.
    """
    with debug_logger() as logger:
        # Baseline test
        cat_image = get_test_image(
            subject="cat",
            size="256x256",
            number=1
        )
        try:
            cat_image_result = get_test_result(
                subject="midas_depth",
                size="256x256",
                number=1
            )
        except FileNotFoundError:
            cat_image_result = None

        test_results = execute_task_test_suite(
            "depth-detection",
            model="midas",
            cases=[
                ({"image": cat_image}, cat_image_result)
            ],
        )

        if cat_image_result is None:
            save_test_image(
                test_results[0],
                "midas_depth"
            )

def test_midas_normal() -> None:
    """
    Test the midas (normal detection) task.
    """
    with debug_logger() as logger:
        # Baseline test
        cat_image = get_test_image(
            subject="cat",
            size="256x256",
            number=1
        )
        try:
            cat_image_result = get_test_result(
                subject="midas_normal",
                size="256x256",
                number=1
            )
        except FileNotFoundError:
            cat_image_result = None

        test_results = execute_task_test_suite(
            "depth-detection",
            model="midas",
            cases=[
                ({"image": cat_image, "mode": "normal"}, cat_image_result)
            ],
        )

        if cat_image_result is None:
            save_test_image(
                test_results[0],
                "midas_normal"
            )

def test_midas_depth_normal() -> None:
    """
    Test the midas (depth + normal detection) task.
    """
    with debug_logger() as logger:
        # Baseline test
        cat_image = get_test_image(
            subject="cat",
            size="256x256",
            number=1
        )
        try:
            cat_depth_result = get_test_result(
                subject="midas_depth",
                size="256x256",
                number=1
            )
            cat_normal_result = get_test_result(
                subject="midas_normal",
                size="256x256",
                number=1
            )
            expected_result = (cat_depth_result, cat_normal_result)
        except FileNotFoundError:
            expected_result = None
        test_results = execute_task_test_suite(
            "depth-detection",
            model="midas",
            cases=[
                (
                    {"image": cat_image, "mode": "depth-normal"},
                    expected_result
                )
            ],
        )
