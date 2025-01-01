from taproot import Task
from taproot.util import (
    debug_logger,
    get_test_image,
)

def test_image_similarity() -> None:
    """
    Test the image similarity task.
    """
    with debug_logger() as logger:
        task_class = Task.get("image-similarity", model=None, available_only=False)
        assert task_class is not None
        cat_image = get_test_image(subject="cat", size="256x256")
        dog_image = get_test_image(subject="dog", size="256x256")
        for method in ["mse", "ssim", "psnr", "histogram", "features"]:
            logger.warning(f"Testing method: {method} (same image)")
            similarity = task_class.exercise(
                left=cat_image,
                right=cat_image,
                method=method,
                num_executions=7
            )
            assert 0.99 <= similarity <= 1.0
            logger.warning(f"Testing method: {method} (different images)")
            similarity = task_class.exercise(
                left=cat_image,
                right=dog_image,
                method=method,
                num_executions=7
            )
            assert 0.0 <= similarity <= 0.80
