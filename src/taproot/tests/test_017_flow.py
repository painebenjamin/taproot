from typing import Any
from tqdm import tqdm
from taproot.util import (
    Video,
    get_test_video,
    save_test_video,
)

def execute_test_flow_method(
    name: str,
    video: Video,
    sparse: bool,
    **kwargs: Any,
) -> None:

    print(f"Testing {name} flow method")

    if sparse:
        frame_iterator = video.sparse_flow_image(**kwargs)
    else:
        assert "method" in kwargs, "method is required"
        frame_iterator = video.dense_flow_image(**kwargs)

    frames = [f for f in tqdm(frame_iterator)]

    save_test_video(
        frames,
        subject=f"flow_{name}",
        frame_rate=video.frame_rate, # type: ignore[arg-type]
    )

def test_flow_methods() -> None:
    """
    Test the flow methods.
    """
    video_path = get_test_video(subject="bmx")
    video = Video.from_file(video_path) # type: ignore[arg-type]

    execute_test_flow_method("sparse", video, sparse=True)
    execute_test_flow_method("dense_lucas_kanade", video, sparse=False, method="dense-lucas-kanade")
    execute_test_flow_method("dense_farneback", video, sparse=False, method="farneback")
    execute_test_flow_method("dense_rlof", video, sparse=False, method="rlof")
