import asyncio
from taproot import Client, Overseer, Dispatcher
from taproot.util import (
    debug_logger,
    generate_id,
    generate_temp_key_and_cert,
    get_metadata,
    get_test_server_protocols,
    get_test_server_addresses,
)

def test_overseer() -> None:
    """
    Test the overseer on it's own, in all deployment configurations.
    """
    with debug_logger() as logger:
        # Create the overseer and dispatcher
        keyfile, certfile = generate_temp_key_and_cert()
        base_encryption_config = {
            "encryption_key": "test" * 8,
            "keyfile": keyfile,
            "certfile": certfile,
        }
        overseer = Overseer({
            "encryption": base_encryption_config,
            "dispatcher_score_timeout": 1.0,
        })
        dispatcher = Dispatcher({
            "encryption": base_encryption_config,
            "executor_config": {
                **{
                    "max_idle_time": 1.0,
                    "encryption": base_encryption_config,
                }
            }
        })

        # Create the client, we'll configure the address later
        client = Client()
        client.certfile = certfile
        client.encryption_key = base_encryption_config["encryption_key"] # type: ignore[assignment]

        payload = {"message": "Hello, World!"}
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
            "parameters": payload
        }

        async def execute_test() -> None:
            """
            Execute the test
            """
            logger.info(f"Testing overseer with dispatcher at {dispatcher.address} and overseer at {overseer.address}, and executor protocol {dispatcher.config.executor_config.protocol}")

            # Run the dispatcher first
            dispatcher_task = asyncio.create_task(dispatcher.run())
            await asyncio.sleep(0.1)

            # Configure and run the overseer
            overseer.unregister_all_dispatchers()
            overseer.register_dispatcher(dispatcher.address)
            overseer_task = asyncio.create_task(overseer.run())
            await asyncio.sleep(0.1)

            # Configure the client
            client.address = overseer.address

            try:
                # First ask to overseer to prepare an executor
                executor_address = await client(metadata_payload, timeout=1.5)
                logger.info(f"Received executor address: {executor_address}")
                executor_client = Client()
                executor_client.address = executor_address["address"]
                executor_client.encryption_key = base_encryption_config["encryption_key"] # type: ignore[assignment]
                executor_client.certfile = certfile
                await asyncio.sleep(0.1)
                # Now execute the payload
                logger.info(f"Sending payload to executor: {full_payload}")
                response = await executor_client(full_payload)
                assert response == "Hello, World!"
                await asyncio.sleep(1.2)
                try:
                    logger.info("Sending payload to executor again, should fail.")
                    await executor_client(full_payload, retries=0)
                    assert False, "Executor should have been stopped"
                except ConnectionError:
                    assert True
            finally:
                # Stop both the dispatcher and overseer
                try:
                    await asyncio.gather(
                        overseer.exit(timeout=1.0),
                        dispatcher.exit(timeout=1.0)
                    )
                except:
                    pass
                dispatcher_task.cancel()
                overseer_task.cancel()
                await asyncio.sleep(0.4)

        for protocol in get_test_server_protocols(no_memory=True):
            dispatcher.config.executor_config.protocol = protocol
            for overseer_address in get_test_server_addresses(no_memory=True):
                overseer.address = overseer_address
                for dispatcher_address in get_test_server_addresses(no_memory=True):
                    dispatcher.address = dispatcher_address
                    asyncio.run(execute_test())
