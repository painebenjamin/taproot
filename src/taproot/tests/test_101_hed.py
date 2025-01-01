from taproot.util import (
    debug_logger,
    get_test_image,
    get_test_images,
    save_test_image,
    get_test_result,
    execute_task_test_suite,
)

def test_hed() -> None:
    """
    Test the HED (holistically-nested edge detection) task.
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
                subject="hed",
                size="256x256",
                number=1
            )
        except FileNotFoundError:
            cat_image_result = None

        test_results = execute_task_test_suite(
            "edge-detection",
            model="hed",
            cases=[
                ({"image": cat_image}, cat_image_result)
            ],
        )
        if cat_image_result is None:
            save_test_image(
                test_results[0],
                "hed"
            )
        # Batch tests
        dog_images = get_test_images(
            num_images=10,
            subject="dog",
            size="256x256"
        )
        for batch_size in 5, 10:
            execute_task_test_suite(
                "edge-detection",
                model="hed",
                cases=[
                    ({"image": dog_images[:batch_size]}, None)
                ]
            )
