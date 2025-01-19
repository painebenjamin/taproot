from taproot.util import (
    debug_logger,
    get_test_image,
    save_test_image,
    get_test_result,
    execute_task_test_suite
)

def test_traditional_super_resolution() -> None:
    """
    Test traditional super-resolution models.
    """
    with debug_logger() as logger:
        # Baseline test
        cat_image = get_test_image(
            subject="cat",
            size="256x256",
            number=1
        )

        for method in ["bicubic", "nearest", "bilinear", "lanczos", "box", "hamming"]:
            try:
                cat_image_result = get_test_result(
                    subject=f"2x_{method}",
                    size="512x512",
                    number=1
                )
            except FileNotFoundError:
                cat_image_result = None

            test_results = execute_task_test_suite(
                "super-resolution",
                cases=[
                    ({"image": cat_image, "method": method}, cat_image_result)
                ]
            )

            if cat_image_result is None:
                save_test_image(
                    test_results[0],
                    f"2x_{method}",
                )
