import asyncio
import logging
from time import perf_counter
from taproot import TaskQueue
from taproot.util import (
    debug_logger,
    save_test_audio,
    execute_task_test_suite,
    human_duration,
    log_duration,
)

def test_kokoro() -> None:
    """
    Test the kokoro model.
    """
    with debug_logger() as logger:
        [hello] = execute_task_test_suite(
            "speech-synthesis",
            model="kokoro",
            assert_runtime_memory_ratio=None,
            num_exercise_executions=3,
            cases=[
                ({"text": "Hello, world!", "seed": 12345, "enhance": False}, None),
            ]
        )
        save_test_audio(
            hello,
            "hello",
            sample_rate=24000
        )
        [goodbye] = execute_task_test_suite(
            "speech-synthesis",
            model="kokoro",
            num_exercise_executions=3,
            assert_runtime_memory_ratio=None,
            assert_static_memory_ratio=None,
            cases=[
                ({"text": "Goodbye, world!", "seed": 12345, "enhance": True}, None),
            ]
        )
        save_test_audio(
            goodbye,
            "goodbye",
            sample_rate=48000
        )
