import asyncio
import logging
from time import perf_counter
from taproot import Task, TaskQueue
from taproot.util import (
    debug_logger,
    save_test_audio,
    execute_task_test_suite,
    human_duration,
    log_duration,
)

def test_xtts2() -> None:
    """
    Test the xtts2 model.
    """
    with debug_logger() as logger:
        [hello] = execute_task_test_suite(
            "speech-synthesis",
            model="xtts-v2",
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
            model="xtts-v2",
            num_exercise_executions=3,
            cases=[
                ({"text": "Goodbye, world!", "seed": 12345, "enhance": True}, None),
            ]
        )
        save_test_audio(
            goodbye,
            "goodbye",
            sample_rate=48000
        )

def test_xtts2_streaming() -> None:
    """
    Test the xtts2 model with streaming.
    """
    with debug_logger() as logger:
        text = """Once upon a midnight dreary, while I pondered, weak and weary,
        Over many a quaint and curious volume of forgotten lore—
        While I nodded, nearly napping, suddenly there came a tapping,
        As of some one gently rapping, rapping at my chamber door.
        “’Tis some visitor,” I muttered, “tapping at my chamber door—
        Only this and nothing more.”"""
        task_class = Task.get("speech-synthesis", model="xtts-v2")
        from taproot.tasks.generation.audio.speech.xtts2.task import XTTS2SpeechSynthesis
        assert task_class is XTTS2SpeechSynthesis, "Could not find task"
        task = XTTS2SpeechSynthesis()
        task.load()
        chunks = []

        with log_duration("warmup"):
            task.xtts(text="Hello, world!", stream=False)

        with log_duration("first chunk"):
            stream = task.xtts(
                text=text,
                stream=True,
                stream_chunk_size=20
            )
            chunks.append(next(stream)) # type: ignore[arg-type]

        while True:
            with log_duration("chunk"):
                try:
                    chunk = next(stream) # type: ignore[arg-type]
                    chunks.append(chunk)
                except StopIteration:
                    break

        import torch
        save_test_audio(
            torch.cat(chunks),
            "xtts2_streaming",
            sample_rate=24000
        )

def test_xtts2_task_streaming() -> None:
    """
    Test the xtts2 model with streaming via the task interface.
    """
    with debug_logger(logging.INFO) as logger:
        text = """Ah, distinctly I remember it was in the bleak December;
        And each separate dying ember writhed upon the floor.
        Eagerly I wished the morrow;—vainly I had sought to borrow
        From my books surcease of sorrow—sorrow for the lost Lenore—
        For the rare and radiant maiden whom the angels name Lenore—
        Nameless here for evermore."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        queue = TaskQueue({
            "task": "speech-synthesis",
            "model": "xtts-v2"
        })
        loop.run_until_complete(queue._wait_for_task())

        with log_duration("warmup"):
            warmup_result = queue(text="Hello, world!", stream=False, enhance=True)
            while warmup_result["status"] not in ["complete", "error"]:
                loop.run_until_complete(asyncio.sleep(0.02))
                warmup_result = queue(id=warmup_result["id"])
            if warmup_result["status"] == "error":
                raise RuntimeError("Warmup failed")

        first_intermediate = True
        start = perf_counter()
        result = queue(text=text, stream=True, enhance=True, output_format="float")
        with log_duration("streaming"):
            while result["status"] not in ["complete", "error"]:
                loop.run_until_complete(asyncio.sleep(0.1))
                result = queue(id=result["id"])
                if result.get("intermediate", None) is not None:
                    if first_intermediate:
                        logger.info(f"First intermediate received in {human_duration(perf_counter() - start)}")
                        first_intermediate = False
                    logger.info(f"Number of samples: {result['intermediate'].shape[0]}")

        save_test_audio(
            result["result"],
            "xtts2_streaming_task",
            sample_rate=48000
        )
        # Clean up
        queue.shutdown()
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()
