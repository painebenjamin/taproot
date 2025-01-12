from taproot import Task
from taproot.util import (
    debug_logger,
    get_test_image,
    execute_task_test_suite,
)

def test_inception_exercise() -> None:
    """
    Tests the inception image similarity task in task exercising context.
    This is used as a baseline AI model inference task, as it is a common task
    that is used in many AI applications and requires a GPU and model.
    """
    with debug_logger() as logger:
        task_class = Task.get("image-similarity", model="inception-v3", available_only=False)
        assert task_class is not None
        cat_image = get_test_image(subject="cat", size="256x256")
        dog_image = get_test_image(subject="dog", size="256x256")
        # Euclidean distance
        euclidean_distance = task_class.exercise(
            left=cat_image,
            right=dog_image,
        )
        assert 0.2 < euclidean_distance < 0.6

def test_inception_suite() -> None:
    """
    Uses the test suite helper.
    """
    with debug_logger() as logger:
        cat_image = get_test_image(subject="cat", size="256x256")
        dog_image = get_test_image(subject="dog", size="256x256")
        execute_task_test_suite(
            "image-similarity",
            model="inception-v3",
            assert_static_memory_ratio=None, # Don't check static memory as it's likely still present
            cases=[
                ({"left": cat_image, "right": dog_image}, None), # Don't check the output
            ]
        )
