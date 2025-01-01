import asyncio
from taproot import Server, Client
from taproot.util import (
    debug_logger,
    get_test_server_addresses,
    log_duration,
    human_duration,
    time_counter
)

def test_server_client() -> None:
    """
    Test the server and client classes.
    """
    import torch
    import numpy as np
    from PIL import Image

    # Test all payload types
    payloads = [
        "Hello, world!",
        4,
        False,
        1.5,
        {"a": 1, "b": 2},
        [1, 2, 3],
        np.random.rand(5, 3),
        Image.new("RGB", (5, 5)),
        torch.rand(256, 256, 8, 4), # Large
    ]

    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    asyncio.set_event_loop(loop)

    with debug_logger() as logger:
        # Create default server and client
        server = Server()
        client = Client()

        # Request to send in all tests
        request = "Hello, world!"

        # Define a test function
        async def execute_test() -> None:
            """
            Execute the test.
            """
            # Start server
            server_task = asyncio.create_task(server.run())
            await asyncio.sleep(0.01)

            # Send request
            client.address = server.address
            client.encryption_key = server.encryption_key
            client.certfile = server.certfile

            try:
                for payload in payloads:
                    with time_counter() as duration:
                        response = await client(payload)
                    logger.info(f"{server.address} - {type(payload).__name__}: {human_duration(float(duration))}")
                    try:
                        # Check response is echoed
                        if isinstance(payload, Image.Image):
                            assert np.array_equal(np.array(payload), np.array(response))
                        elif isinstance(payload, torch.Tensor):
                            assert torch.equal(payload, response)
                        elif isinstance(payload, np.ndarray):
                            assert np.array_equal(payload, response)
                        else:
                            assert response == payload
                    except AssertionError as e:
                        raise AssertionError(f"Test failed for server address {server.address}") from e
            finally:
                # Stop server
                server_task.cancel()
                await asyncio.sleep(0.01)

        async def execute_timeout_test() -> None:
            """
            Execute the timeout test.
            """
            server.max_idle_time = 0.2
            server_task = asyncio.create_task(server.run())
            await asyncio.sleep(0.01)
            with log_duration(f"{server.address}"):
                assert await client(request) == request
            await asyncio.sleep(0.3)
            try:
                await client(request, retries=0) == request
                assert False, "Server should have timed out"
            except ConnectionError:
                assert True
            finally:
                server_task.cancel()
                await asyncio.sleep(0.01)

        async def execute_shutdown_test() -> None:
            """
            Executes a shutdown test.
            """
            server.use_control_encryption = True
            try:
                server_task = asyncio.create_task(server.run())
                await asyncio.sleep(0.01)
                await server.assert_connectivity()
                with log_duration(f"{server.address}"):
                    try:
                        no_encryption = await client("control:shutdown")
                        assert False, "Server should have rejected shutdown command."
                    except Exception as e:
                        with_encryption = await client(server.pack_control_message("shutdown"))
                        assert True
            finally:
                server.use_control_encryption = False
                server_task.cancel()
                await asyncio.sleep(0.01)

        for server_address in get_test_server_addresses():
            server.address = server_address
            loop.run_until_complete(execute_test())
            loop.run_until_complete(execute_timeout_test())
            loop.run_until_complete(execute_shutdown_test())
