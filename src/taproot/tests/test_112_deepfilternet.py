from taproot.util import (
    debug_logger,
    get_test_audio,
    save_test_audio,
    execute_task_test_suite,
)

def test_deep_filter_net() -> None:
    # Test the deepfilternet3 model on the audio file "a-time-for-choosing".
    with debug_logger() as logger:
        short_audio = get_test_audio(subject="polly")
        long_audio = get_test_audio(subject="maya-angelou")
        [polly, maya_angelou] = execute_task_test_suite(
            "speech-enhancement",
            model="deep-filter-net-v3",
            num_exercise_executions=3,
            assert_runtime_memory_ratio=None, # unreliable
            cases=[
                ({"audio": short_audio}, None),
                ({"audio": long_audio}, None)
            ]
        )
        save_test_audio(
            maya_angelou,
            "dfn3_maya_angelou.mp3",
            sample_rate=42000
        )
