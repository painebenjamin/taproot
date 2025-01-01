from taproot.util import (
    debug_logger,
    execute_task_test_suite,
    get_test_image,
    save_test_video
)

from typing import Any

def run_video_generation_test_suite(
    model_name: str,
    **kwargs: Any
) -> None:
    """
    Runs a video generation test suite for the specified model.
    """
    with debug_logger() as logger:
        logger.info(f"Running image generation test suite for model: {model_name}")
        prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
        seed = 123456
        [result] = execute_task_test_suite(
            "video-generation",
            model=model_name,
            assert_static_memory_ratio=None,
            assert_runtime_memory_ratio=None,
            num_exercise_executions=1,
            cases = [
                (
                    {
                        **{
                            "prompt": prompt,
                            "seed": seed,
                            "output_format": "png",
                        },
                        **kwargs
                    },
                    None
                ),
            ]
        )
        save_test_video(
            result,
            model_name,
            frame_rate=kwargs.get("frame_rate", 8),
        )

def test_ltx() -> None:
    """
    Tests the LTX model for video generation.
    """
    run_video_generation_test_suite("ltx")

def test_ltx_int8() -> None:
    """
    Tests the LTX model for video generation with int-8 quantization.
    """
    run_video_generation_test_suite("ltx-int8")

def test_ltx_nf4() -> None:
    """
    Tests the LTX model for video generation with NF4 quantization.
    """
    run_video_generation_test_suite("ltx-nf4")

def test_mochi() -> None:
    """
    Tests the Mochi V1 Preview model for video generation
    """
    run_video_generation_test_suite("mochi-v1")

def test_mochi_int8() -> None:
    """
    Tests the Mochi V1 Preview model for video generation with int8 quantization
    """
    run_video_generation_test_suite("mochi-v1-int8")

def test_mochi_nf4() -> None:
    """
    Tests the Mochi V1 Preview model for video generation with nf4 quantization
    """
    run_video_generation_test_suite("mochi-v1-nf4")

def test_cogvideox_2b() -> None:
    """
    Tests the CogVideoX 2B model for video generation
    """
    run_video_generation_test_suite("cogvideox-2b")

def test_cogvideox_2b_int8() -> None:
    """
    Tests the CogVideoX 2B model for video generation with int8 quantization
    """
    run_video_generation_test_suite("cogvideox-2b-int8")

def test_cogvideox_5b() -> None:
    """
    Tests the CogVideoX 5B model for video generation
    """
    run_video_generation_test_suite("cogvideox-5b")

def test_cogvideox_5b_int8() -> None:
    """
    Tests the CogVideoX 5B model for video generation with int8 quantization
    """
    run_video_generation_test_suite("cogvideox-5b-int8")

def test_cogvideox_5b_nf4() -> None:
    """
    Tests the CogVideoX 5B model for video generation with nf4 quantization
    """
    run_video_generation_test_suite("cogvideox-5b-nf4")

def test_cogvideox_i2v_5b() -> None:
    """
    Tests the CogVideoX 5B model for image-to-video generation
    """
    run_video_generation_test_suite(
        "cogvideox-i2v-5b",
        image=get_test_image(
            size="720x480",
            subject="waves"
        ),
        prompt="Waves splash against a sheer rock face during a stormy day. The water is a deep blue, and the sky is overcast with dark clouds. The waves are large and powerful, crashing against the rocks with a loud roar. The scene is dramatic and intense, with the waves reaching high into the air before crashing back down. The rocks are wet and glistening with water, and the air is filled with the smell of salt and seaweed."
    )

def test_cogvideox_v15_5b() -> None:
    """
    Tests the CogVideoX 5B model for video generation
    """
    run_video_generation_test_suite("cogvideox-v1-5-5b")

# Probably not worth it to run this in int8
def test_cogvideox_v15_5b_int8() -> None:
    """
    Tests the CogVideoX 5B model for video generation
    """
    run_video_generation_test_suite("cogvideox-v1-5-5b-int8")

def test_cogvideox_v15_5b_nf4() -> None:
    """
    Tests the CogVideoX 5B model for video generation
    """
    run_video_generation_test_suite("cogvideox-v1-5-5b-nf4")

def test_cogvideox_v15_i2v_5b() -> None:
    """
    Tests the CogVideoX 5B model for image-to-video generation
    """
    run_video_generation_test_suite(
        "cogvideox-v1-5-i2v-5b",
        image=get_test_image(
            size="1360x768",
            subject="waves"
        ),
        prompt="Waves splash against a sheer rock face during a stormy day. The water is a deep blue, and the sky is overcast with dark clouds. The waves are large and powerful, crashing against the rocks with a loud roar. The scene is dramatic and intense, with the waves reaching high into the air before crashing back down. The rocks are wet and glistening with water, and the air is filled with the smell of salt and seaweed.",
        num_frames=25
    )

def test_cogvideox_v15_i2v_5b_int8() -> None:
    """
    Tests the CogVideoX 5B model for image-to-video generation
    """
    run_video_generation_test_suite(
        "cogvideox-v1-5-i2v-5b-int8",
        image=get_test_image(
            size="1360x768",
            subject="waves"
        ),
        prompt="Waves splash against a sheer rock face during a stormy day. The water is a deep blue, and the sky is overcast with dark clouds. The waves are large and powerful, crashing against the rocks with a loud roar. The scene is dramatic and intense, with the waves reaching high into the air before crashing back down. The rocks are wet and glistening with water, and the air is filled with the smell of salt and seaweed.",
        num_frames=17
    )

def test_cogvideox_v15_i2v_5b_nf4() -> None:
    """
    Tests the CogVideoX 5B model for image-to-video generation
    """
    run_video_generation_test_suite(
        "cogvideox-v1-5-i2v-5b-nf4",
        image=get_test_image(
            size="1360x768",
            subject="waves"
        ),
        prompt="Waves splash against a sheer rock face during a stormy day. The water is a deep blue, and the sky is overcast with dark clouds. The waves are large and powerful, crashing against the rocks with a loud roar. The scene is dramatic and intense, with the waves reaching high into the air before crashing back down. The rocks are wet and glistening with water, and the air is filled with the smell of salt and seaweed.",
        num_frames=17
    )
