from taproot.util import (
    debug_logger,
    get_test_image,
    save_test_image,
    get_test_result,
    execute_task_test_suite
)

def test_aura() -> None:
    """
    Test the aura super-resolution task.
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
                subject="4x_aura",
                size="1024x1024",
                number=1
            )
        except FileNotFoundError:
            cat_image_result = None

        test_results = execute_task_test_suite(
            "super-resolution",
            model="aura",
            cases=[
                ({"image": cat_image, "seed": 12345}, cat_image_result)
            ]
        )
        if cat_image_result is None:
            save_test_image(
                test_results[0],
                "4x_aura"
            )
