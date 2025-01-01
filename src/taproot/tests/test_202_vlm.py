from typing import Optional
from taproot.util import debug_logger, get_test_image, execute_task_test_suite

def execute_vlm_suite(model: Optional[str] = None) -> None:
    """
    Execute the test suite for the language model task.
    """
    with debug_logger() as logger:
        image = get_test_image(
            size="512x512",
            subject="house"
        )
        seed = 12345
        kwargs = {
            "prompt": "Please describe the main subject of this image in a few words.",
            "image": image,
            "seed": seed,
            "max_tokens": 32,
        }
        [completed] = execute_task_test_suite(
            "visual-question-answering",
            assert_static_memory_ratio=0.1, # Ensure +/- 10%
            assert_runtime_memory_ratio=None, # We only care about static (for now)
            model=model,
            cases=[
                (kwargs, None)
            ]
        )
        assert isinstance(completed, str), "Expected completion to be a string"
        assert "error" not in completed.lower(), "Expected no error in completion"
        if "house" not in completed.lower():
            logger.warning(f"Expected 'house' in completion. This isn't necessarily a failure, but is unexpected. Result was: {completed}")

# TODO: JoyCaption

def test_moondream() -> None:
    execute_vlm_suite("moondream-v2")

def test_llava_v15_7b() -> None:
    execute_vlm_suite("llava-v1-5-7b")

def test_llava_v15_7b_q8() -> None:
    execute_vlm_suite("llava-v1-5-7b-q8")

def test_llava_v15_7b_q6_k() -> None:
    execute_vlm_suite("llava-v1-5-7b-q6-k")

def test_llava_v15_7b_q5_k_m() -> None:
    execute_vlm_suite("llava-v1-5-7b-q5-k-m")

def test_llava_v15_7b_q4_k_m() -> None:
    execute_vlm_suite("llava-v1-5-7b-q4-k-m")

def test_llava_v15_7b_q3_k_m() -> None:
    execute_vlm_suite("llava-v1-5-7b-q3-k-m")

def test_llava_v15_13b() -> None:
    execute_vlm_suite("llava-v1-5-13b")

def test_llava_v15_13b_q6_k() -> None:
    execute_vlm_suite("llava-v1-5-13b-q6-k")

def test_llava_v15_13b_q5_k_m() -> None:
    execute_vlm_suite("llava-v1-5-13b-q5-k-m")

def test_llava_v15_13b_q4_0() -> None:
    execute_vlm_suite("llava-v1-5-13b-q4-0")

# TODO: More v1.6 models and quants

def test_llava_v16_34b_q5_k_m() -> None:
    execute_vlm_suite("llava-v1-6-34b-q5-k-m")

def test_llava_v16_34b_q4_k_m() -> None:
    execute_vlm_suite("llava-v1-6-34b-q4-k-m")

def test_llava_v16_34b_q3_k_m() -> None:
    execute_vlm_suite("llava-v1-6-34b-q3-k-m")
