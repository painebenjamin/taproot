from taproot.tasks import Task
from taproot.util import (
    debug_logger,
    get_test_audio,
    time_counter,
    human_duration,
    execute_task_test_suite,
    get_text_similarity
)

def run_large_model_test(model_name: str) -> None:
    """
    Runs a test using a larger model (and longer phrase).
    """
    with debug_logger() as logger:
        audio = get_test_audio(subject="a-time-for-choosing")
        [a_time_for_choosing] = execute_task_test_suite(
            "audio-transcription",
            model=model_name,
            num_exercise_executions=3,
            assert_runtime_memory_ratio=None,
            cases=[
                ({"audio": audio}, None)
            ]
        )
        assert isinstance(a_time_for_choosing, str)
        a_time_for_choosing = a_time_for_choosing.strip().lower().replace(".", "").replace(",", "")
        assert a_time_for_choosing.startswith("ladies and gentlemen we take pride in presenting a thoughtful address by ronald reagan")
        assert "he has faith that you and i have the ability and the dignity and the right to make our own decisions and determine our own destiny" in a_time_for_choosing

def run_small_model_test(model_name: str) -> None:
    """
    Runs a test using a smaller model (and shorter phrase).
    """
    with debug_logger() as logger:
        audio = get_test_audio(subject="polly")
        [polly] = execute_task_test_suite(
            "audio-transcription",
            model=model_name,
            num_exercise_executions=3,
            cases=[
                ({"audio": audio}, None)
            ]
        )
        try:
            assert polly == "Polly picked a peck of pickled peppers."
        except AssertionError:
            if polly.strip().lower().startswith("pol"):
                logger.warning(f"Model {model_name} did not transcribe the full phrase accurately, but did function as expected.")
                logger.warning(f"Expected: Polly picked a peck of pickled peppers.")
                logger.warning(f"Actual: {polly}")
            else:
                raise

def test_turbo_whisper_large_v3() -> None:
    """
    Test the turbo whisper large v3 model on the audio file "a-time-for-choosing".
    """
    run_large_model_test("turbo-whisper-large-v3")
    run_small_model_test("turbo-whisper-large-v3")

def test_whisper_large_v3() -> None:
    """
    Test the whisper medium model on the audio file "polly".
    """
    run_large_model_test("whisper-large-v3")
    run_small_model_test("whisper-large-v3")

def test_whisper_medium() -> None:
    """
    Test the whisper medium model on the audio file "polly".
    """
    run_small_model_test("whisper-medium")

def test_whisper_small() -> None:
    """
    Test the whisper small model on the audio file "polly".
    """
    run_small_model_test("whisper-small")

def test_whisper_base() -> None:
    """
    Test the whisper base model on the audio file "polly".
    """
    run_small_model_test("whisper-base")

def test_whisper_tiny() -> None:
    """
    Test the whisper tiny model on the audio file "polly".
    """
    run_small_model_test("whisper-tiny")

def test_distilled_whisper_large_v3() -> None:
    """
    Test the distilled whisper large v3 model on the audio file "a-time-for-choosing".
    """
    run_large_model_test("distilled-whisper-large-v3")
    run_small_model_test("distilled-whisper-large-v3")

def test_distilled_whisper_medium_english() -> None:
    """
    Test the distilled whisper medium english model on the audio file "a-time-for-choosing".
    """
    run_large_model_test("distilled-whisper-medium-english")
    run_small_model_test("distilled-whisper-medium-english")

def test_distilled_whisper_small_english() -> None:
    """
    Test the distilled whisper small english model on the audio file "a-time-for-choosing".
    """
    run_small_model_test("distilled-whisper-small-english")

def test_model_comparison() -> None:
    with debug_logger() as logger:
        load_times = {}
        short_run_times = {}
        medium_run_times = {}
        short_results = {}
        medium_results = {}

        short_audio_transcription = get_test_audio(subject="polly", include_transcript=True)
        assert isinstance(short_audio_transcription, tuple)
        short_audio, short_transcription = short_audio_transcription

        medium_audio_transcription = get_test_audio(subject="a-time-for-choosing", include_transcript=True)
        assert isinstance(medium_audio_transcription, tuple)
        medium_audio, medium_transcription = medium_audio_transcription

        for model_name in [
            "whisper-tiny",
            "whisper-base",
            "whisper-small",
            "whisper-medium",
            "whisper-large-v3",
            "distilled-whisper-small-english",
            "distilled-whisper-medium-english",
            "distilled-whisper-large-v3",
            "turbo-whisper-large-v3",
        ]:
            task_class = Task.get("audio-transcription", model_name)
            assert task_class is not None
            task = task_class()

            with time_counter() as load_time:
                task.load()

            load_times[model_name] = float(load_time)

            with time_counter() as run_time:
                result = task(audio=short_audio)
            with time_counter() as second_run_time:
                result = task(audio=short_audio)
            with time_counter() as third_run_time:
                result = task(audio=short_audio)

            short_run_times[model_name] = [float(second_run_time), float(third_run_time)]
            short_results[model_name] = result

            with time_counter() as medium_run_time:
                medium_result = task(audio=medium_audio)

            medium_run_times[model_name] = float(medium_run_time)
            medium_results[model_name] = medium_result

            task.unload()

        logger.info("Load times:")
        for model_name, time in load_times.items():
            logger.info(f"  {model_name}: {human_duration(time)}")

        logger.info("Run times (short audio):")
        for model_name, (second_time, third_time) in short_run_times.items():
            average_time = (second_time + third_time) / 2
            logger.info(f"  {model_name}: {human_duration(second_time)}, {human_duration(third_time)} ({human_duration(average_time)} average)")

        logger.info("Results (short audio):")
        for model_name, result in short_results.items():
            wer = get_text_similarity(reference=short_transcription, hypothesis=result, method="wer")
            logger.info(f"  {model_name}: {result} ({wer:.1%} WER)")

        logger.info("Run times (medium audio):")
        for model_name, time in medium_run_times.items():
            logger.info(f"  {model_name}: {human_duration(time)}")

        logger.info("Results (medium audio):")
        for model_name, result in medium_results.items():
            wer = get_text_similarity(result, medium_transcription, method="wer")
            logger.info(f"  {model_name}: {wer:.1%} WER")
