from taproot.util import (
    debug_logger,
    get_test_image,
    get_test_images,
    save_test_image,
    get_test_result,
    execute_task_test_suite,
)

def test_anime_lineart() -> None:
    """
    Test the anime lineart detection task.
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
                subject="anime_lineart",
                size="256x256",
                number=1
            )
        except FileNotFoundError:
            cat_image_result = None

        test_results = execute_task_test_suite(
            "line-detection",
            model="informative-drawings-anime",
            cases=[
                ({"image": cat_image}, cat_image_result)
            ],
        )

        if cat_image_result is None:
            save_test_image(
                test_results[0],
                "anime_lineart"
            )

def run_batch(num_images: int) -> None:
    """
    Test with a batch size.
    """
    with debug_logger() as logger:
        dog_images = get_test_images(
            num_images=num_images,
            subject="dog",
            size="256x256"
        )
        execute_task_test_suite(
            "line-detection",
            model="informative-drawings-anime",
            cases=[
                ({"image": dog_images}, None)
            ]
        )

def test_batch_5() -> None:
    run_batch(5)

def test_batch_10() -> None:
    run_batch(10)

def test_batch_50() -> None:
    run_batch(50)