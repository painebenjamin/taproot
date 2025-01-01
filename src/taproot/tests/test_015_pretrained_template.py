from taproot.util import (
    debug_logger,
    get_test_image,
)

def test_pretrained_template() -> None:
    """
    Tests the pretrained template.
    This is not in the global namespace so we test it explicitly.
    """
    with debug_logger() as logger:
        from taproot.tasks.template import TaskTemplate
        from PIL import Image
        dog_image = get_test_image(subject="dog", size="256x256")
        result = TaskTemplate.exercise(image=dog_image) # Will install torch and download the model if needed
        assert isinstance(result, Image.Image)
