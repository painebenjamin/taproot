from taproot.util import (
    debug_logger,
    get_test_image,
    get_test_images,
    save_test_image,
    get_test_result,
    execute_task_test_suite,
)

def test_mlsd() -> None:
    """
    Test the mlsd task.
    """
    with debug_logger() as logger:
        # Baseline test
        house_image = get_test_image(
            subject="house",
            size="512x512",
            number=1
        )
        try:
            house_image_result = get_test_result(
                subject="mlsd",
                size="512x512",
                number=1
            )
        except FileNotFoundError:
            house_image_result = None

        test_results = execute_task_test_suite(
            "line-detection",
            model="mlsd",
            cases=[
                ({"image": house_image}, house_image_result)
            ],
        )

        if house_image_result is None:
            save_test_image(
                test_results[0],
                "mlsd"
            )

def run_batch(num_images: int) -> None:
    """
    Test with a batch size.
    """
    with debug_logger() as logger:
        dog_images = get_test_images(
            num_images=num_images,
            subject="house",
            size="512x512"
        )
        execute_task_test_suite(
            "line-detection",
            model="mlsd",
            cases=[
                ({"image": dog_images}, None)
            ]
        )

def test_batch_5() -> None:
    run_batch(5)

def test_batch_10() -> None:
    run_batch(10)
