import asyncio
import logging

from typing import Any, Dict
from taproot import Client, Dispatcher
from taproot.util import (
    debug_logger,
    time_counter,
    get_metadata,
    find_free_port,
    get_test_server_protocols,
    get_test_server_addresses,
)

def test_dispatcher() -> None:
    """
    Test the dispatcher on it's own, in all deployment configurations.
    """
    with debug_logger() as logger:
        # Create the dispatcher, we'll configure the address later
        dispatcher = Dispatcher({
            "spawn_interval": None,
            "max_workers": 3,
            "use_multiprocessing": True,
            "executor_config": {
                "max_idle_time": 0.4,
            },
        })

        # Create the client, we'll configure the address later
        client = Client()
        parameters = {"message": 3, "delay": 1.0}
        parameter_metadata = get_metadata(parameters)

        # Payload will be manipulated as needed
        payload = {
            "task": "echo",
            "client_id": "test",
            "parameters": parameters,
        }

        async def execute_test() -> None:
            """
            Execute the test
            """
            # Run the dispatcher
            logger.info(f"Running the dispatcher at at {dispatcher.address} with {dispatcher.config.executor_config.protocol} protocol")
            task = asyncio.create_task(dispatcher.run())
            await asyncio.sleep(0.01)

            # Configure the client
            client.address = dispatcher.address
            client.encryption_key = dispatcher.encryption_key
            client.use_encryption = dispatcher.use_encryption
            client.certfile = dispatcher.certfile

            try:
                # First ask to score the task on the dispatcher
                payload["parameters"] = parameter_metadata

                score = await client.command("score", data=payload, timeout=0.2)
                assert 0 < score <= 10000, f"Score should be between 0 and 10000, got {score} on {dispatcher.address}"

                # Now ask for the dispatcher to prepare the executor
                num_requests = 3
                executor_address_payloads = await asyncio.gather(*[
                    client(
                        client.pack_control_message("prepare",data=payload),
                        timeout=.2
                    )
                    for i in range(num_requests)
                ])
                logger.info(f"Got {len(executor_address_payloads)} executors")
                logger.debug(executor_address_payloads)

                executor_addresses = [p["address"] for p in executor_address_payloads]
                request_ids = [p["id"] for p in executor_address_payloads]
                await asyncio.sleep(0.01)

                executor_clients = [Client() for i in range(num_requests)]
                for address, executor_client in zip(executor_addresses, executor_clients):
                    executor_client.address = address
                    executor_client.encryption_key = dispatcher.encryption_key
                    executor_client.certfile = dispatcher.certfile

                payload["parameters"] = parameters
                with time_counter() as timer:
                    response = await asyncio.gather(*[
                        executor_client({
                            **payload,
                            **{"id": request_id}
                        })
                        for executor_client, request_id in zip(executor_clients, request_ids)
                    ])

                # The sleep is 1 second, a successful parallel execution will be less than 2 seconds.
                # A failed parallel execution will be one second per request.
                assert timer < 2
                assert response == [3] * num_requests
                logger.info("Waiting for executor to timeout")
                await asyncio.sleep(2.0) # Wait for the executor to timeout
                try:
                    result = await executor_clients[0](payload, retries=0)
                    assert False, f"Executor {executor_clients[0].address} should have been stopped"
                except ConnectionError:
                    assert True
            finally:
                # Stop the dispatcher
                try:
                    logger.info("Stopping the dispatcher")
                    dispatcher.graceful_exit()
                    await asyncio.sleep(0.2)
                except Exception as e:
                    logger.warning(f"Error stopping the dispatcher: {e}")

        for protocol in get_test_server_protocols(no_memory=True):
            dispatcher.config.executor_config.protocol = protocol
            for address in get_test_server_addresses():
                logger.info(f"Testing dispatcher at {address} with {protocol} protocol")
                dispatcher.address = address
                try:
                    asyncio.run(execute_test())
                except:
                    logger.critical(f"Failing test for {dispatcher.address} with {dispatcher.config.executor_config.protocol} protocol")
                    raise

def _test_configured_executors() -> None:
    """
    Tests a dispatcher with a preconfigured number of executors.
    """
    with debug_logger(logging.INFO) as logger:
        # Create the dispatcher
        dispatcher = Dispatcher({
            "protocol": "tcp",
            "host": "127.0.0.1",
            "port": find_free_port(),
            "max_workers": 5,
            "use_multiprocessing": True,
            "executor_config": {
                "max_idle_time": 1.0,
                "protocol": "tcp",
                "queue_config": {
                    "size": 5
                }
            },
            "task_max_workers": {
                "echo": 3
            },
            "static_executor_config": [
                {   
                    "queue_config": {
                        "task": "echo",
                    },
                    "protocol": "tcp",
                    "host": "127.0.0.1",
                    "port": find_free_port(),
                },
                {
                    "queue_config": {
                        "task": "echo",
                    },
                    "protocol": "tcp",
                    "host": "127.0.0.1",
                    "port": find_free_port(),
                }
            ]
        })

        async def execute_test() -> None:
            # Start the dispatcher
            task = asyncio.create_task(dispatcher.run())
            start = asyncio.get_event_loop().time()
            while True:
                await asyncio.sleep(0.01)
                status = await dispatcher.status()
                if len(status["executors"].get("echo", [])) == dispatcher.config.static_executor_config:
                    logger.info(f"Executors started after {asyncio.get_event_loop().time() - start:.2f} seconds")
                    break
                if asyncio.get_event_loop().time() - start > 4:
                    assert False, "Timeout waiting for executors to start"

            payload = {
                "task": "echo",
                "client_id": "test",
                "parameters": {},
            }

            num_payloads = 0

            def make_payload() -> Dict[str, Any]:
                nonlocal num_payloads
                num_payloads += 1
                return {
                    **payload,
                    **{
                        "id": f"test_{num_payloads}",
                        "parameters": get_metadata({"message": f"{num_payloads}", "delay": 1.0})
                    }
                }

            client = dispatcher.get_client()

            # Get the maximum number of addresses
            await asyncio.gather(*[
                client.command(
                    "prepare",
                    data=make_payload(),
                    timeout=1.0
                )
                for i in range(dispatcher.config.task_max_workers["echo"])
            ])

            # Try and get one more, this should fail
            try:
                await client.command(
                    "prepare",
                    data=make_payload(),
                    retries=0,
                    timeout=0.3
                )
                assert False, "Should have failed to get another executor"
            except Exception as e:
                if isinstance(e, AssertionError):
                    raise e
                assert True

            # Exit the dispatcher
            dispatcher.graceful_exit()
            await asyncio.sleep(0.01)
            task.cancel()

        # Run the test
        asyncio.run(execute_test())
