from taproot import Client, Overseer, Dispatcher
from taproot.util import (
    debug_logger,
    generate_id,
    get_metadata,
    find_free_port,
    time_counter,
    AsyncRunner,
)

def test_continuation() -> None:
    """
    Tests continuations without using the Tap helper.
    Doesn't exercise all the possible combinations of protocols and addresses.
    """
    async def execute_test() -> None:
        with debug_logger() as logger:
            # Create the overseer and dispatcher
            overseer = Overseer()
            overseer.address = f"tcp://127.0.0.1:{find_free_port()}"
            dispatcher = Dispatcher()
            dispatcher.address = f"tcp://127.0.0.1:{find_free_port()}"
            dispatcher.executor_protocol = "tcp"
            dispatcher.max_workers = 2

            # Configure payloads
            payload = {
                "message": "Hello, World!",
                "delay": 0.5
            }
            payload_id = generate_id()
            metadata_payload = {
                "task": "echo",
                "id": payload_id,
                "client_id": "test",
                "parameters": get_metadata(payload)
            }
            full_payload = {
                "task": "echo",
                "id": payload_id,
                "client_id": "test",
                "parameters": payload,
                "overseer": overseer.address,
                "return_metadata": True,
                "continuation": {
                    "task": "echo",
                    "result_parameters": "message",
                    "parameters": {
                        "delay": 0.5
                    }
                }
            }
            async with overseer:
                async with dispatcher:
                    await dispatcher.register_overseer(overseer.address)
                    # Request an executor
                    executor_payload = await overseer.get_client()(metadata_payload)
                    logger.info(f"Executor payload: {executor_payload}")

                    # Send the payload to the executor
                    executor_client = Client()
                    executor_client.address = executor_payload["address"]

                    # Wait for the first result
                    with time_counter() as first_execution:
                        result = await executor_client(full_payload)

                    # Check the result
                    assert 0.0 < float(first_execution) < 0.75
                    assert result.get("continuation", None) is not None

                    # Request the continuation
                    continuation_payload = {
                        "id": result["continuation"]["id"],
                        "client_id": "test"
                    }

                    with time_counter() as second_execution:
                        second_executor_client = Client()
                        second_executor_client.address = result["continuation"]["address"]
                        continuation_result = await second_executor_client(continuation_payload)

                    # Check the continuation result
                    assert 0.0 < float(second_execution) < 0.75
                    assert continuation_result == payload["message"]

                    # Try to request too many continuations
                    full_payload["id"] = generate_id()
                    current = full_payload["continuation"]
                    for i in range(11):
                        current["continuation"] = {**current} # type: ignore[index, dict-item]
                        current = current["continuation"] # type: ignore[index]

                    with time_counter() as third_execution:
                        result = await executor_client(full_payload)

                    assert 0.0 < float(third_execution) < 0.75
                    assert isinstance(result.get("continuation", None), str)
                    assert result["continuation"].startswith("Error")

                    await dispatcher.exit()
                    await overseer.exit()

    AsyncRunner(execute_test).run(debug=True)
