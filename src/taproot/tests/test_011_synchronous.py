from taproot import Tap
from taproot.util import debug_logger, log_duration, time_counter

def test_synchronous() -> None:
    """
    Tests simple synchronous usage with 'call' and 'parallel' syntax.
    """
    with debug_logger() as logger:
        with Tap.local(
            use_multiprocessing=True,
            task_auto_executors={"echo": 5},
        ) as tap:
            with log_duration("executor initialization"):
                assert tap.call("echo", message="Hello!", delay=0.0) == "Hello!"
            with time_counter() as duration:
                assert tap.parallel(
                    {
                        "task": "echo",
                        "parameters": {"message": "Hello!", "delay": 1.0}
                    },
                    {
                        "task": "echo",
                        "parameters": {"message": "World!", "delay": 1.0}
                    },
                    {
                        "task": "echo",
                        "parameters": {"message": "Goodbye!", "delay": 1.0}
                    },
                    {
                        "task": "echo",
                        "parameters": {"message": "World!", "delay": 1.0}
                    },
                ) == ["Hello!", "World!", "Goodbye!", "World!"]
            assert duration < 2.5
