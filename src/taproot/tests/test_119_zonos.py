from taproot.util import (
    debug_logger,
    save_test_audio,
    execute_task_test_suite,
)

def test_zonos_hybrid() -> None:
    """
    Test the zonos hybrid model.
    """
    with debug_logger() as logger:
        [hello] = execute_task_test_suite(
            "speech-synthesis",
            model="zonos-hybrid",
            assert_runtime_memory_ratio=None,
            num_exercise_executions=3,
            cases=[
                ({"text": "Hello, world!", "seed": 12345, "enhance": False}, None),
            ]
        )
        save_test_audio(
            hello,
            "zonos_hybrid_hello",
            sample_rate=44000
        )
        [goodbye] = execute_task_test_suite(
            "speech-synthesis",
            model="zonos-hybrid",
            num_exercise_executions=3,
            assert_runtime_memory_ratio=None,
            assert_static_memory_ratio=None,
            cases=[
                ({"text": "Goodbye, world!", "seed": 12345, "enhance": True}, None),
            ]
        )
        save_test_audio(
            goodbye,
            "zonos_hybrid_goodbye",
            sample_rate=48000
        )

def test_zonos_transformer() -> None:
    """
    Test the zonos transformer model.
    """
    with debug_logger() as logger:
        [hello] = execute_task_test_suite(
            "speech-synthesis",
            model="zonos-transformer",
            assert_runtime_memory_ratio=None,
            assert_static_memory_ratio=None,
            num_exercise_executions=3,
            cases=[
                ({"text": "Hello, world!", "seed": 12345, "enhance": False}, None),
            ]
        )
        save_test_audio(
            hello,
            "zonos_transformer_hello",
            sample_rate=44000
        )
        [goodbye] = execute_task_test_suite(
            "speech-synthesis",
            model="zonos-transformer",
            num_exercise_executions=3,
            assert_runtime_memory_ratio=None,
            assert_static_memory_ratio=None,
            cases=[
                ({"text": "Goodbye, world!", "seed": 12345, "enhance": True}, None),
            ]
        )
        save_test_audio(
            goodbye,
            "zonos_transformer_goodbye",
            sample_rate=48000
        )
