import asyncio
from taproot import Tap, Task, Overseer, Dispatcher
from taproot.util import debug_logger, log_duration, find_free_port

def test_tap_continuation() -> None:
    """
    Test the tap using continuations.
    """
    # Define a task for the test
    class Square(Task):
        task = "square"
        def __call__(self, *, arg: int) -> int: # type: ignore[override]
            return arg * arg

    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    asyncio.set_event_loop(loop)

    with debug_logger() as logger:
        overseer = Overseer()
        overseer.address = f"tcp://127.0.0.1:{find_free_port()}"

        dispatcher = Dispatcher()
        dispatcher.address = f"tcp://127.0.0.1:{find_free_port()}"
        dispatcher.executor_protocol = "tcp"
        dispatcher.spawn_interval = 0.0 # Disable interval
        dispatcher.max_workers = 2 # Need at least 2 for recursion

        overseer_task = loop.create_task(overseer.run())
        dispatcher_task = loop.create_task(dispatcher.run())
        
        loop.run_until_complete(dispatcher.register_overseer(overseer.address))

        tap = Tap()
        tap.remote_address = overseer.address

        sanity = asyncio.run(tap("square", arg=2))
        assert sanity == 4

        # Use one continuation
        with log_duration("one_continuation"):
            continued = asyncio.run(
                tap("square", arg=2, continuation={"task": "square", "result_parameters": "arg"})
            )
            logger.info(f"Continued: {continued}")

        with log_duration("two_continuation"):
            continued = asyncio.run(
                tap("square", arg=2, continuation={"task": "square", "result_parameters": "arg", "continuation": {"task": "square", "result_parameters": "arg"}})
            )
            logger.info(f"Continued: {continued}")

        loop.run_until_complete(dispatcher.exit())
        loop.run_until_complete(overseer.exit())
