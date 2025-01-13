from __future__ import annotations

import os
import sys
import click
import signal
import asyncio

from random import randint, randbytes
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Type,
    TypeVar
)
from contextlib import contextmanager, nullcontext

from ..constants import *

from .log_util import debug_logger, logger
from .system_util import catch_output
from .async_util import ServerRunner, aioconsole_is_available
from .string_util import human_size, human_duration
from .test_util import time_counter
from .terminal_util import (
    maybe_use_tqdm,
    magenta,
    green,
    cyan,
    red
)

if TYPE_CHECKING:
    from ..server import Server, ConfigServer
    T = TypeVar("T", bound=ConfigServer)
else:
    T = TypeVar("T")

__all__ = [
    "get_command_context",
    "get_pidfile_context",
    "get_line_count",
    "context_options",
    "server_options",
    "get_server",
    "get_server_runner",
    "run_echo_test",
    "async_chat_loop"
]

@contextmanager
def get_command_context(
    log_level: str,
    add_import: List[str]=[],
    quiet: bool=False
) -> Iterator[None]:
    """
    Get command context.
    """
    if quiet:
        output_context = catch_output(ignore_output=True, reset_on_exit=False)
    else:
        output_context = nullcontext() # type: ignore[assignment]

    with output_context:
        with debug_logger(log_level.upper()) as logger:
            for import_str in add_import:
                logger.info(f"Importing {import_str}.")
                __import__(import_str)
            yield

@contextmanager
def get_pidfile_context(
    pidfile: Optional[str],
    exclusive: bool=False
) -> Iterator[None]:
    """
    Get PID file context.
    """
    if exclusive and pidfile and os.path.exists(pidfile):
        # Read PID file
        pid = open(pidfile, "r").read().strip()
        # Check if PID is running
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            # Not running / no process with PID
            pass
        except PermissionError:
            # Running as root, but PID is not accessible
            click.echo("PID file exists and process is running. Exiting.")
            sys.exit(1)
        else:
            # Running as user, PID is accessible
            click.echo("PID file exists and process is running. Exiting.")
            sys.exit(1)

    if pidfile:
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))

    yield

    if pidfile:
        try:
            os.remove(pidfile)
        except:
            pass

def get_line_count(text: str) -> int:
    """
    Gets the number of lines in a text after wrapping.
    """
    lines = text.strip().split("\n")
    num_lines = 0
    width = os.get_terminal_size().columns
    for line in lines:
        line_len = len(line)
        if line_len <= width:
            num_lines += 1
        else:
            num_lines += line_len // width + 1
    return num_lines

def context_options(
    include_log_levels: bool=True,
    include_add_import: bool=True,
    include_quiet: bool=True,
    include_model_dir: bool=True,
    include_save_dir: bool=False,
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """
    Decorator for context options.
    """
    def wrapper(func: Callable[..., None]) -> Callable[..., None]:
        """
        Decorator for context options.
        """
        if include_model_dir:
            func = click.option(
                "--model-dir",
                "-m",
                type=str,
                default=DEFAULT_MODEL_DIR,
                help="Model directory.",
                show_default=True
            )(func)

        if include_save_dir:
            func = click.option(
                "--save-dir",
                "-s",
                type=str,
                default=DEFAULT_SAVE_DIR,
                help="Directory to save output to.",
                show_default=True
            )(func)

        if include_log_levels:
            log_levels = ["debug", "info", "warning", "error", "critical"]
            for log_level in log_levels:
                func = click.option(
                    f"--{log_level}",
                    "log_level",
                    flag_value=log_level,
                    default=log_level.upper() == DEFAULT_LOG_LEVEL,
                    show_default=True,
                    help=f"Set log level to {log_level}."
                )(func)
        
        if include_add_import:
            func = click.option(
                "--add-import",
                multiple=True,
                type=str,
                help="Additional imports. Use this to add custom tasks, roles, tools, etc."
            )(func)

        if include_quiet:
            func = click.option(
                "--quiet",
                "-q",
                is_flag=True,
                help="Suppress stdout and stderr."
            )(func)

        return func
    return wrapper

def server_options(
    default_address: str=DEFAULT_ADDRESS,
    include_address: bool=True,
    include_config_file: bool=True,
    include_pidfile: bool=True,
    include_certificates: bool=True,
    include_control_encryption_key: bool=True,
    include_lists: bool=True,
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """
    Decorator for server options.
    """
    def wrapper(func: Callable[..., None]) -> Callable[..., None]:
        """
        Decorator for server options.
        """
        if include_address:
            func = click.argument(
                "address",
                type=str,
                default=default_address,
                required=False,
            )(func)

        if include_certificates:
            func = click.option(
                "--certfile",
                "-cf",
                type=click.Path(exists=True),
                default=None,
                help="SSL certificate file when using WSS.",
                show_default=True
            )(func)

            func = click.option(
                "--keyfile",
                "-kf",
                type=click.Path(exists=True),
                default=None,
                help="SSL key file when using WSS.",
                show_default=True
            )(func)

            func = click.option(
                "--cafile",
                "-caf",
                type=click.Path(exists=True),
                default=None,
                help="SSL CA file when using WSS.",
                show_default=True
            )(func)
    
        if include_control_encryption_key:
            func = click.option(
                "--control-encryption-key",
                "-cek",
                type=str,
                default=None,
                help="Encryption key for control messages.",
                show_default=True
            )(func)

        if include_pidfile:
            func = click.option(
                "--pidfile",
                "-p",
                type=click.Path(),
                default=None,
                help="PID file to write to.",
                show_default=True
            )(func)
            
            func = click.option(
                "--exclusive",
                "-e",
                type=bool,
                default=False,
                show_default=True,
                is_flag=True,
                help="Exclusively run one overseer (requires --pidfile)."
            )(func)

        if include_lists:
            func = click.option(
                "--allow",
                "-a",
                type=str,
                multiple=True,
                help="Allow list for all tcp connections.",
                show_default=True
            )(func)

            func = click.option(
                "--reject",
                "-r",
                type=str,
                multiple=True,
                help="Reject list for all tcp connections.",
                show_default=True
            )(func)

            func = click.option(
                "--allow-control",
                "-ac",
                type=str,
                multiple=True,
                help="Allow list for control connections (remote lifecycle management).",
                show_default=True
            )(func)

        if include_config_file:
            func = click.option(
                "--config",
                "-c",
                type=click.Path(exists=True),
                default=None,
                help="Configuration file.",
                show_default=True
            )(func)

        return func
    return wrapper

def get_server(
    server_class: Type[T],
    address: str=DEFAULT_ADDRESS,
    default_address: str=DEFAULT_ADDRESS,
    config: Optional[str]=None,
    certfile: Optional[str]=None,
    keyfile: Optional[str]=None,
    cafile: Optional[str]=None,
    allow: List[str]=[],
    reject: List[str]=[],
    allow_control: List[str]=[],
    control_encryption_key: Optional[str]=None,
) -> T:
    """
    Gets a server instance.
    """
    server = server_class(config)
    if config is None or address != default_address:
        server.address = address
    if certfile is not None:
        server.certfile = certfile
    if keyfile is not None:
        server.keyfile = keyfile
    if cafile is not None:
        server.cafile = cafile
    if allow:
        server.allow_list = list(allow)
    if allow_control:
        server.control_list = list(allow_control)
    if control_encryption_key is not None:
        server.use_control_encryption = True
        server.control_encryption_key = control_encryption_key # type: ignore[assignment]
    return server

def get_server_runner(*servers: Server) -> ServerRunner:
    """
    Gets a server runner.
    """
    runner = ServerRunner(*servers)
    server_labels = ", ".join(
        f"{type(server).__name__} at {server.address}"
        for server in servers
    )
    runner.before_start(lambda: click.echo(f"Starting {server_labels}."))
    runner.after_start(lambda: click.echo(f"Monitoring {server_labels}."))
    runner.before_stop(lambda: click.echo(f"Stopping {server_labels}."))
    runner.after_stop(lambda: click.echo(f"Stopped {server_labels}."))
    return runner

async def run_echo_test(
    address: str,
    packet_size_bytes: Tuple[int, ...] = (1, 1_000, 1_000_000, 10_000_000),
    num_packets_per_size: int = 3,
    use_tqdm: bool = True
) -> None:
    """
    Runs a test echo client.
    """
    from ..client import Client
    client = Client()
    client.address = address

    # Generate test data
    test_packets = [
        [randbytes(packet_size) for _ in range(num_packets_per_size)]
        for packet_size in packet_size_bytes
    ]

    # Send data in striped packages (e.g. 1 byte, 1KB, 1MB, 1 byte, 1KB, 1MB, ...)
    test_times: List[List[float]] = []
    for i in maybe_use_tqdm(range(num_packets_per_size), use_tqdm=use_tqdm, desc="Iteration"):
        test_times.append([])
        for j, packet_list in enumerate(test_packets):
            with time_counter() as timer:
                result = await client(packet_list[i])
            assert result == packet_list[i], "Echo test failed!"
            test_times[-1].append(float(timer))

    # Transpose
    test_times = list(zip(*test_times)) # type: ignore[arg-type]

    # Calculate average times
    average_times = [sum(times) / len(times) for times in test_times]

    # Print results
    click.echo("Echo test results:")
    for i, packet_size in enumerate(packet_size_bytes):
        click.echo(f"Packet size: {magenta(human_size(packet_size))}")
        for j in range(num_packets_per_size):
            click.echo(f"Packet {j + 1}: {green(human_duration(test_times[i][j]))}")
        click.echo(f"Average time: {green(human_duration(average_times[i]))}")
        click.echo(f"Transfer rate: {cyan(human_size(packet_size * 2 / average_times[i]))}/s")
        click.echo()

async def async_chat_loop(
    model: Optional[str]=None,
    model_dir: str=DEFAULT_MODEL_DIR,
    forgetful: bool=False,
    stream: bool=False,
    role: Optional[str]=None,
    seed: Optional[int]=None,
    use_tools: bool=False,
    max_tokens: Optional[int]=None,
    context_length: Optional[int]=None,
    log_level: str=DEFAULT_LOG_LEVEL,
    add_import: List[str]=[]
) -> None:
    """
    The main chat loop.
    """
    if not aioconsole_is_available():
        raise RuntimeError("aioconsole is not available. Please install aioconsole with 'pip install aioconsole' to use the chat loop.")

    with get_command_context(log_level, add_import):
        from aioconsole import ainput # type: ignore[import-untyped]
        from ..tasks import TaskQueue

        queue = TaskQueue.get(
            "text-generation",
            model=model,
            task_config={
                "context_length": context_length
            }
        )
        queue.start()
        await queue.wait_for_task()
        conversation: List[str] = []

        if seed is None:
            seed = randint(0x10000000, 0xFFFFFFFF)

        system = magenta("[system]")
        assistant = cyan("[assistant]")
        user = green("[user]")
        brk = "\033[K"

        click.echo(f"{system} Model set to {queue._task.model}.{brk}")
        click.echo(f"{system} Seed set to {seed}.{brk}")

        exit_event = asyncio.Event()

        def signal_handler(signum: int, frame: Any) -> None:
            exit_event.set()
            raise KeyboardInterrupt()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while not exit_event.is_set():
                prompt = await ainput(f"{user} {brk}")
                if not prompt:
                    continue
                if forgetful:
                    conversation = []
                if prompt.lower() in ["reset", "forget", "clear"]:
                    conversation = []
                    click.echo(f"{system} Context cleared.{brk}")
                    continue
                elif prompt.lower() in ["exit", "quit", "bye", "goodbye"]:
                    raise EOFError
                elif prompt.lower().startswith("role:"):
                    conversation = []
                    role = prompt.split(":", 1)[1].strip().lower()
                    if role == "none":
                        role = None
                    click.echo(f"{system} Role set to {role}.{brk}")
                    continue
                elif prompt.lower().startswith("seed:"):
                    conversation = []
                    seed_str = prompt.split(":", 1)[1].strip()
                    if seed_str.lower() in ["random", "rand"]:
                        seed = randint(0x10000000, 0xFFFFFFFF)
                    else:
                        try:
                            seed = int(seed_str)
                        except ValueError:
                            click.echo(f"{system} Invalid seed value.{brk}")
                            continue
                    click.echo(f"{system} Seed set to {seed}.{brk}")
                    continue
                conversation.append(prompt)
                result = queue(
                    prompt=conversation,
                    role=role,
                    seed=seed,
                    stream=stream,
                    max_tokens=max_tokens,
                    use_tools=use_tools
                )
                num_lines = 0
                skipped_first_clear = False
                clear_lines: Callable[[], int] = lambda: sys.stdout.write("\033[F\033[K" * (num_lines - 1))

                while result["status"] not in ["complete", "error"]:
                    if result.get("intermediate", None):
                        temporary_response_text = f"{assistant} {result['intermediate']}"
                        if num_lines == 2 and not skipped_first_clear:
                            skipped_first_clear = True
                            if "\n" in temporary_response_text:
                                clear_lines()
                        else:
                            clear_lines()
                        sys.stdout.write(f"\r{temporary_response_text}{brk}")
                        sys.stdout.flush()
                        this_num_lines = get_line_count(temporary_response_text)
                        if this_num_lines > num_lines:
                            num_lines = this_num_lines
                    await asyncio.sleep(0.05)
                    result = queue(id=result["id"])

                if result["status"] == "complete":
                    response_text = f"{assistant} {result['result']}"
                    conversation.append(result["result"])
                    clear_lines()
                    click.echo(f"\r{response_text}{brk}")
                else:
                    error_text = red(result["result"] or "error")
                    clear_lines()
                    click.echo(f"\r{assistant} {error_text}{brk}")
        except (EOFError, KeyboardInterrupt, asyncio.exceptions.CancelledError) as ex:
            pass

        await queue.shutdown()
        click.echo(f"\r{system} Goodbye!{brk}")
        loop = asyncio.get_event_loop()
        for task in asyncio.all_tasks(loop):
            task.cancel()
