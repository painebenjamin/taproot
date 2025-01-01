import asyncio
from taproot import Tap, Overseer, Dispatcher
from taproot.constants import *
from taproot.config import *
from taproot.util import (
    debug_logger,
    log_duration,
    generate_temp_key_and_cert,
    get_test_server_protocols,
    get_test_server_addresses,
    get_test_restricted_import_context,
    find_free_memory_port,
    find_free_port,
    find_free_unix_socket,
)

def test_tap() -> None:
    """
    Test the tap on it's own, in all deployment configurations.
    """
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    with get_test_restricted_import_context(): # Ensure some modules are never initialized
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
                "dispatcher_score_timeout": 0.4,
            })
            dispatcher = Dispatcher({
                "encryption": base_encryption_config,
                "executor_config": {
                    "max_idle_time": 0.4,
                    "encryption": base_encryption_config,
                }
            })

            tap = Tap()

            async def execute_test() -> None:
                """
                Execute the test
                """
                # Find a free host/port
                if dispatcher.protocol == "memory":
                    dispatcher.port = find_free_memory_port()
                elif dispatcher.protocol in ["tcp", "ws"]:
                    dispatcher.host = DEFAULT_HOST
                    dispatcher.port = find_free_port()
                else:
                    dispatcher.path = find_free_unix_socket()

                # Run the dispatcher
                dispatcher_task = loop.create_task(dispatcher.run())
                await asyncio.sleep(0.05)
                await dispatcher.assert_connectivity()

                # Register the dispatcher
                overseer.unregister_all_dispatchers()
                overseer.register_dispatcher(dispatcher.address)

                # Execute the test
                with log_duration("base"):
                    assert await tap("echo", message="Hello!") == "Hello!"

                # Gracefully exit the dispatcher
                if not dispatcher.graceful_exit():
                    dispatcher_task.cancel()

                await asyncio.sleep(0.5)

            for overseer_address in get_test_server_addresses(no_memory=True):
                # Set overseer address
                overseer.address = overseer_address

                # Run the overseer
                overseer_task = loop.create_task(overseer.run())
                loop.run_until_complete(overseer.assert_connectivity())

                # Configure the tap
                tap.remote_address = overseer.address
                tap.remote_encryption_key = overseer.encryption_key
                tap.remote_certfile = overseer.certfile

                for dispatcher_address in get_test_server_addresses():
                    # Set dispatcher address
                    dispatcher.address = dispatcher_address

                    # Configure the dispatcher
                    dispatcher.config.executor_config.encryption = None

                    # Test all protocols
                    for protocol in get_test_server_protocols(no_memory=True):
                        dispatcher.config.executor_config.protocol = protocol
                        logger.info(f"Testing overseer={overseer_address}, dispatcher={dispatcher_address}, protocol={protocol}")
                        loop.run_until_complete(execute_test())

                    # Now ensure this works when the executor is encrypted,
                    # regardless of whether or not the dispatcher is encrypted.
                    dispatcher.config.executor_config.protocol = "tcp"
                    dispatcher.config.executor_config.encryption = EncryptionConfig(
                        encryption_key=overseer.encryption_key.decode("utf-8"),
                    )
                    logger.info(f"Testing encrypted overseer={overseer_address}, dispatcher={dispatcher_address}, protocol=tcp")
                    loop.run_until_complete(execute_test())

                logger.info(f"Exiting overseer={overseer_address}")
                if not overseer.graceful_exit():
                    logger.error("Failed to gracefully exit overseer")
                    overseer_task.cancel()
                    loop.run_until_complete(asyncio.sleep(0.2))
    # Clean up
    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    loop.close()
