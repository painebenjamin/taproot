import asyncio
from taproot import Tap
from taproot.util import debug_logger, asyncio_run, time_counter

def test_parallel() -> None:
    """
    Tests running multiple tasks in parallel
    """
    with debug_logger() as logger:
        with Tap.local(
            use_multiprocessing=True,
            max_workers=10
        ) as tap:
            async def execute_test() -> None:
                """
                Execute the test
                """
                messages = list(range(1,11))
                with time_counter() as duration:
                    tasks = [
                        tap("echo", delay=1.0, message=m)
                        for m in messages
                    ]
                    results = await asyncio.gather(*tasks)
                assert results == messages
                # Why 4 seconds when it should be paralellizing up to 10 tasks at once, so in theory
                # this should only take 1 second plus a small overhead? The reason is since we're on a 
                # single node, the initial acquisition of an executor will clash with other tasks launched
                # simulatenously that also need an executor. The reservation clash takes a retry to resolve,
                # which is why it takes a bit longer. On a distributed system, this is not an issue, and only
                # occurs during initial launch of the first tasks.
                assert duration <= 4.0
                logger.info(f"First duration: {duration}")
                # Do it again
                with time_counter() as second_duration:
                    tasks = [
                        tap("echo", delay=1.0, message=m)
                        for m in messages
                    ]
                    results = await asyncio.gather(*tasks)
                assert results == messages
                # The second duration will likely be less than 3 seconds, since
                # most of the results are cached, but there will likely still
                # be at least one clash which will result in a delay of AT LEAST 1 second,
                # based on the simulated task duration.
                assert second_duration <= 3.0
                logger.info(f"Second duration: {second_duration}")

            asyncio_run(execute_test())
