from taproot.util import (
    debug_logger,
    get_test_audio,
    save_test_audio,
    execute_task_test_suite,
)

def test_f5tts() -> None:
    """
    Test the f5tts model.
    """
    long_text = "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences."
    really_long_text = "Have you ever heard the tragedy of Darth Playgiss the Wise? I thought not. It's not a story the Jed-I would tell you. It's a Sith legend. Darth Playgiss was a Dark Lord of the Sith, so powerful and so wise he could use the Force to influence the midichlorians to create life. He had such a knowledge of the dark side that he could even keep the ones he cared about from dying. The dark side of the Force is a pathway to many abilities some consider to be unnatural. He became so powerful the only thing he was afraid of was losing his power, which eventually, of course, he did. Unfortunately, he taught his apprentice everything he knew, then his apprentice killed him in his sleep. It's ironic he could save others from death, but not himself."
    with debug_logger() as logger:
        # Get default audio and transcript from f5tts repo
        default_audio_transcript = get_test_audio(subject="f5tts", include_transcript=True)
        assert isinstance(default_audio_transcript, tuple)
        default_audio, default_transcript = default_audio_transcript

        # Get our test audio and transcript from dipco
        dipco_audio_transcript = get_test_audio(subject="dipco", include_transcript=True)
        assert isinstance(dipco_audio_transcript, tuple)
        dipco_audio, dipco_transcript = dipco_audio_transcript

        # Get last test audio which will be transcribed
        transatlantic = get_test_audio(subject="transatlantic")

        for name, audio, transcript in [
            ("f5tts", default_audio, default_transcript),
            ("dipco", dipco_audio, dipco_transcript),
            ("transatlantic", transatlantic, None)
        ]:
            kwargs = {
                "reference_audio": audio,
                "reference_text": transcript,
                "seed": 12345,
            }
            if transcript is not None:
                [hello, repeat, repeat_enhance, long, long_enhance, really_long, really_long_enhance] = execute_task_test_suite(
                    "speech-synthesis",
                    model="f5tts",
                    num_exercise_executions=1,
                    assert_runtime_memory_ratio=None, # Unreliable
                    cases=[
                        ({"text": "Hello, world!", "enhance": False, **kwargs}, None),
                        ({"text": transcript, "enhance": False, **kwargs}, None),
                        ({"text": transcript, "enhance": True, **kwargs}, None),
                        ({"text": long_text, "enhance": False, **kwargs}, None),
                        ({"text": long_text, "enhance": True, **kwargs}, None),
                        ({"text": really_long_text, "enhance": False, **kwargs}, None),
                        ({"text": really_long_text, "enhance": True, **kwargs}, None),
                    ]
                )
            else:
                [hello, long, long_enhance, really_long, really_long_enhance] = execute_task_test_suite(
                    "speech-synthesis",
                    model="f5tts",
                    num_exercise_executions=1,
                    assert_runtime_memory_ratio=None, # Unreliable
                    cases=[
                        ({"text": "Hello, world!", "enhance": False, **kwargs}, None),
                        ({"text": long_text, "enhance": False, **kwargs}, None),
                        ({"text": long_text, "enhance": True, **kwargs}, None),
                        ({"text": really_long_text, "enhance": False, **kwargs}, None),
                        ({"text": really_long_text, "enhance": True, **kwargs}, None),
                    ]
                )

            save_test_audio(
                hello,
                f"helloworld_{name}",
                sample_rate=24000
            )
            save_test_audio(
                long,
                f"long_{name}",
                sample_rate=24000
            )
            save_test_audio(
                long_enhance,
                f"long_enhance_{name}",
                sample_rate=48000
            )
            save_test_audio(
                really_long,
                f"really_long_{name}",
                sample_rate=24000
            )
            save_test_audio(
                really_long_enhance,
                f"really_long_enhance_{name}",
                sample_rate=48000
            )
            if transcript is not None:
                save_test_audio(
                    repeat,
                    f"repeat_{name}",
                    sample_rate=24000
                )
                save_test_audio(
                    repeat_enhance,
                    f"repeat_enhance_{name}",
                    sample_rate=48000
                )
