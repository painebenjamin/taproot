# type: ignore
import asyncio
from taproot import Client, Executor
from taproot.util import (
    generate_id,
    debug_logger,
    log_duration,
    get_test_server_addresses,
)

def test_executor() -> None:
    """
    Test the executor on it's own, in all deployment configurations.
    """
    with debug_logger() as logger:
        # Create the executor, we'll configure the address later
        executor = Executor({
            "queue_config": {
                "task": "echo",
            }
        })

        # Create the client, we'll configure the address later
        client = Client()

        # This is the payload for all tests
        parameters = {
            "message": "Hello, World!",
            "delay": 0.0
        }
        task_payload: Dict[str, Any] = {
            "task": "echo",
            "parameters": parameters,
            "id": None,
            "client_id": "test"
        }

        async def execute_test() -> None:
            """
            Execute the test
            """
            # Run the executor
            task = asyncio.create_task(executor.run())
            await asyncio.sleep(0.01)

            # Configure the client
            client.address = executor.address
            client.encryption_key = executor.encryption_key
            client.certfile = executor.certfile

            # Send the task payloads
            try:
                # Base test
                task_payload["parameters"]["delay"] = 0.0
                task_payload["id"] = generate_id()
                with log_duration("Base test"):
                    assert await client(task_payload) == "Hello, World!"

                # Parallel test
                task_payload["parameters"]["delay"] = 1.0
                task_payload["id"] = generate_id()
                with log_duration("Parallel test"):
                    result = await asyncio.gather(
                        client(task_payload),
                        client(task_payload),
                    )
                assert result == ["Hello, World!", "Hello, World!"]

                # Do it again to ensure the result is cached
                with log_duration("Parallel test"):
                    result = await asyncio.gather(
                        client(task_payload),
                        client(task_payload),
                    )
                assert result == ["Hello, World!", "Hello, World!"]

                # Ensure activity tracking is working
                status = await client.command("status", data=f"{task_payload['client_id']}:{task_payload['id']}")
                logger.debug(f"Received status: {status}")
                assert status["activity"] > 0.3
                assert status["queued"] == 0

                # Could still be active if it the queue hasn't ticked since last call
                assert status["status"] in ["active", "idle"]
                assert status["has_id"], f"Expected an id in the status: {status}"

                # Finally test reported capability
                capability = await client.command("capability")
                logger.debug(f"Received capability: {capability}")
            except Exception as e:
                logger.error(f"Failed on address {executor.address}")
                raise e
            finally:
                await client.command("exit")
                # Stop the executor
                task.cancel()
                await asyncio.sleep(0.01)

        for address in get_test_server_addresses():
            executor.address = address
            asyncio.run(execute_test())
