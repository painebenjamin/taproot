import asyncio
from taproot import Server, Client
from taproot.util import (
    debug_logger,
    generate_temp_key_and_cert,
    get_test_server_addresses,
    log_duration,
    human_duration,
    time_counter,
    find_free_port,
    AsyncRunner,
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

    with debug_logger() as logger:
        # Create default server and client
        server = Server()
        client = Client()

        keyfile, certfile = generate_temp_key_and_cert()
        server.keyfile = keyfile
        server.certfile = certfile
        client.certfile = server.certfile
        client.encryption_key = server.encryption_key

        # Request to send in all tests
        request = "Hello, world!"

        # Define a test function
        async def execute_test() -> None:
            """
            Execute the test.
            """
            # Start server
            if server.protocol == "http":
                server.port = find_free_port()
            logger.info(f"Beginning test for server address {server.address}")
            async with server:
                # Send request
                client.address = server.address

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

        async def execute_timeout_test() -> None:
            """
            Execute the timeout test.
            """
            # Start server
            if server.protocol == "http":
                server.port = find_free_port()
                client.port = server.port
            logger.info(f"Beginning timeout test for server address {server.address}")
            server.max_idle_time = 0.2
            async with server:
                await asyncio.sleep(0.01)
                with log_duration(f"{server.address}"):
                    assert await client(request) == request
                await asyncio.sleep(0.3)
                try:
                    await client(request, retries=0)
                    assert False, "Server should have timed out"
                except ConnectionError:
                    assert True

        async def execute_shutdown_test() -> None:
            """
            Executes a shutdown test.
            """
            # Start server
            if server.protocol == "http":
                server.port = find_free_port()
                client.port = server.port
            logger.info(f"Beginning shutdown test for server address {server.address}")
            server.use_control_encryption = True
            try:
                async with server:
                    await asyncio.sleep(0.01)
                    await server.assert_connectivity()
                    with log_duration(f"{server.address}"):
                        try:
                            logger.info("Issuing shutdown server, an exception should be raised.")
                            no_encryption = await client("control:shutdown", retries=0)
                            assert False, "Server should have rejected shutdown command."
                        except Exception as e:
                            logger.info(f"Succesfully rejected shutdown command: {e}")
                            with_encryption = await client(server.pack_control_message("shutdown"))
                            assert True
            finally:
                server.use_control_encryption = False
                await asyncio.sleep(0.01)

        for server_address in get_test_server_addresses():
            server.address = server_address
            AsyncRunner(
                execute_test,
                execute_timeout_test,
                execute_shutdown_test,
                sequential=True,
            ).run(debug=True)
