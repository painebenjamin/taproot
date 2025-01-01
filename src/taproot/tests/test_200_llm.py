from typing import Optional
from taproot.util import (
    debug_logger,
    execute_task_test_suite
)

def execute_llm_suite(model: Optional[str] = None) -> None:
    """
    Execute the test suite for the language model task.
    """
    with debug_logger() as logger:
        seed = 12345
        kwargs = {
            "prompt": "Please complete the sentence: 'The quick brown fox jumps over the...'",
            "seed": seed,
            "max_tokens": 32,
        }
        [completed] = execute_task_test_suite(
            "text-generation",
            task_config={"options": {"context_length": 8192}},
            assert_static_memory_ratio=0.1, # Ensure +/- 10%
            assert_runtime_memory_ratio=None, # We only care about static (for now)
            model=model,
            cases=[
                (kwargs, None)
            ]
        )
        assert isinstance(completed, str), "Expected completion to be a string"
        assert "error" not in completed.lower(), "Expected no error in completion"
        if "lazy" not in completed.lower():
            logger.warning(f"Expected 'lazy' in completion. This isn't necessarily a failure, but is unexpected. Result was: {completed}")

def test_zephyr_alpha() -> None:
    execute_llm_suite("zephyr-7b-alpha")

def test_zephyr_alpha_q6_k() -> None:
    execute_llm_suite("zephyr-7b-alpha-q6-k")

def test_zephyr_alpha_q5_k_m() -> None:
    execute_llm_suite("zephyr-7b-alpha-q5-k-m")

def test_zephyr_alpha_q4_k_m() -> None:
    execute_llm_suite("zephyr-7b-alpha-q4-k-m")

def test_zephyr_alpha_q3_k_m() -> None:
    execute_llm_suite("zephyr-7b-alpha-q3-k-m")

def test_zephyr_beta() -> None:
    execute_llm_suite("zephyr-7b-beta")

def test_zephyr_beta_q6_k() -> None:
    execute_llm_suite("zephyr-7b-beta-q6-k")

def test_zephyr_beta_q5_k_m() -> None:
    execute_llm_suite("zephyr-7b-beta-q5-k-m")

def test_zephyr_beta_q4_k_m() -> None:
    execute_llm_suite("zephyr-7b-beta-q4-k-m")

def test_zephyr_beta_q3_k_m() -> None:
    execute_llm_suite("zephyr-7b-beta-q3-k-m")

def test_llama30() -> None:
    execute_llm_suite("llama-v3-8b")

def test_llama30_q6_k() -> None:
    execute_llm_suite("llama-v3-8b-q6-k")

def test_llama30_q5_k_m() -> None:
    execute_llm_suite("llama-v3-8b-q5-k-m")

def test_llama30_q4_k_m() -> None:
    execute_llm_suite("llama-v3-8b-q4-k-m")

def test_llama30_q3_k_m() -> None:
    execute_llm_suite("llama-v3-8b-q3-k-m")

def test_llama30_instruct() -> None:
    execute_llm_suite("llama-v3-8b-instruct")

def test_llama30_instruct_q6_k() -> None:
    execute_llm_suite("llama-v3-8b-instruct-q6-k")

def test_llama30_instruct_q5_k_m() -> None:
    execute_llm_suite("llama-v3-8b-instruct-q5-k-m")

def test_llama30_instruct_q4_k_m() -> None:
    execute_llm_suite("llama-v3-8b-instruct-q4-k-m")

def test_llama30_instruct_q3_k_m() -> None:
    execute_llm_suite("llama-v3-8b-instruct-q3-k-m")

def test_llama31_instruct() -> None:
    execute_llm_suite("llama-v3-1-8b-instruct")

def test_llama31_instruct_q6_k() -> None:
    execute_llm_suite("llama-v3-1-8b-instruct-q6-k")

def test_llama31_instruct_q5_k_m() -> None:
    execute_llm_suite("llama-v3-1-8b-instruct-q5-k-m")

def test_llama31_instruct_q4_k_m() -> None:
    execute_llm_suite("llama-v3-1-8b-instruct-q4-k-m")

def test_llama31_instruct_q3_k_m() -> None:
    execute_llm_suite("llama-v3-1-8b-instruct-q3-k-m")

def test_llama32_instruct() -> None:
    execute_llm_suite("llama-v3-2-3b-instruct")

def test_llama32_instruct_q8_0() -> None:
    execute_llm_suite("llama-v3-2-3b-instruct-q8-0")

def test_llama32_instruct_q6_k() -> None:
    execute_llm_suite("llama-v3-2-3b-instruct-q6-k")

def test_llama32_instruct_q5_k_m() -> None:
    execute_llm_suite("llama-v3-2-3b-instruct-q5-k-m")

def test_llama32_instruct_q4_k_m() -> None:
    execute_llm_suite("llama-v3-2-3b-instruct-q4-k-m")

def test_llama32_instruct_q3_k_l() -> None:
    execute_llm_suite("llama-v3-2-3b-instruct-q3-k-l")

def test_llama32_1b_instruct() -> None:
    execute_llm_suite("llama-v3-2-1b-instruct")

def test_llama32_1b_instruct_q8_0() -> None:
    execute_llm_suite("llama-v3-2-1b-instruct-q8-0")

def test_llama32_1b_instruct_q6_k() -> None:
    execute_llm_suite("llama-v3-2-1b-instruct-q6-k")

def test_llama32_1b_instruct_q5_k_m() -> None:
    execute_llm_suite("llama-v3-2-1b-instruct-q5-k-m")

def test_llama32_1b_instruct_q4_k_m() -> None:
    execute_llm_suite("llama-v3-2-1b-instruct-q4-k-m")

def test_llama32_1b_instruct_q3_k_l() -> None:
    execute_llm_suite("llama-v3-2-1b-instruct-q3-k-l")
