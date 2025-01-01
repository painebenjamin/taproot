from taproot.util import (
    debug_logger,
    get_test_images,
    save_test_video,
    execute_task_test_suite,
)

def test_rife() -> None:
    num_frames = 6
    with debug_logger() as logger:
        [start, end] = get_test_images(
            subject="interpolation",
            size="512x768",
            num_images=2
        )
        kwargs = {
            "start": start,
            "end": end,
            "num_frames": num_frames,
            "include_ends": True
        }
        [frames] = execute_task_test_suite(
            "image-interpolation",
            model="rife",
            num_exercise_executions=3,
            cases=[
                (kwargs, None),
            ]
        )
        save_test_video(
            frames=frames,
            subject="interpolation_rife",
            frame_rate=8,
            format="gif"
        )
