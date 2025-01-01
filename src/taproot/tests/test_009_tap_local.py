import asyncio
from taproot import Tap
from taproot.util import debug_logger, log_duration

def test_tap() -> None:
    """
    Test the tap using a local dispatcher.
    """
    # Use the local dispatcher
    with debug_logger() as logger:
        with Tap.local() as tap:
            # We run a few echo tasks
            # These should be running synchronously. The first will take ~1s (due to spin-up time for server
            # and polling time for machine capabilities), the rest should be faster (about 30ms).
            # Remember these are running in sub-processes.
            for _ in range(10):
                with log_duration("echo"):
                    assert asyncio.run(
                        tap("echo", message="Hello, World!")
                    ) == "Hello, World!"
