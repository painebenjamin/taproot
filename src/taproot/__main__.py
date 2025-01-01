from __future__ import annotations

import os
import re
import sys
import click
import asyncio
import traceback

from contextlib import contextmanager

from time import perf_counter
from random import randint
from typing import Dict, List, Optional, Type, Any, Iterator, Callable

from .version import version
from .constants import *

DEFAULT_OUTPUT_DIR = os.getcwd()

@contextmanager
def get_command_context(
    log_level: str,
    add_import: List[str]=[]
) -> Iterator[None]:
    """
    Get command context.
    """
    from .util import debug_logger
    with debug_logger(log_level.upper()) as logger:
        for import_str in add_import:
            logger.info(f"Importing {import_str}.")
            __import__(import_str)
        yield

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

def log_level_options(func: Callable[..., None]) -> Callable[..., None]:
    """
    Decorator for log level options.
    """
    log_levels = ["debug", "info", "warning", "error", "critical"]
    for log_level in log_levels:
        func = click.option(
            f"--{log_level}",
            "log_level",
            flag_value=log_level,
            default=log_level == "warning",
            show_default=True,
            help=f"Set log level to {log_level}."
        )(func)

    return func

def trim_html_whitespace(html: str) -> str:
    """
    Trims excess whitespace in HTML.
    """
    html = re.sub(r">\s+<", "><", html)
    html = re.sub(r"\s{2,}", " ", html)
    html = re.sub(r"(?<=>)\s+|\s+(?=<)", "", html)
    return html

def context_options(func: Callable[..., None]) -> Callable[..., None]:
    """
    Decorator for context options.
    """
    func = click.option(
        "--add-import",
        multiple=True,
        type=str,
        help="Additional imports. Use this to add custom tasks, roles, tools, etc."
    )(func)

    func = log_level_options(func)

    return func

@click.group(name="taproot")
@click.version_option(version=str(version), message="%(version)s")
def main() -> None:
    """
    Taproot command-line tools.
    """
    pass

@main.command(name="machine-capability", short_help="Print machine capability.")
@context_options
def machine_capability(
    log_level: str="warning",
    add_import: List[str]=[]
) -> None:
    """
    Print machine capability.
    """
    with get_command_context(log_level, add_import):
        from .util import MachineCapability
        capability = MachineCapability.get_capability(fail_on_gpu_error=False)
        click.echo(capability)

@main.command(name="catalog", short_help="Print catalog of available tasks.")
def catalog() -> None:
    """
    Prints complete catalog of available in tabular format.
    """
    import tabulate
    from .tasks import Task
    from .util import get_file_name_from_url, get_file_size_from_url, human_size
    catalog = Task.catalog(available_only = False)
    num_tasks = len(catalog)
    num_models = 0
    for task_name in catalog:
        num_models += len(catalog[task_name]["models"])
    click.echo("<h1>Task Catalog</h1>")
    click.echo(f"<p>{num_tasks} tasks available with {num_models} models.</p>")
    click.echo("<ul>")
    for task_name in catalog:
        num_models = len(catalog[task_name]["models"])
        click.echo(f"<li><a href=\"#{task_name}\">{task_name}</a>: {num_models} model{'s' if num_models != 1 else ''}</li>")
    click.echo("</ul>")
    for task_name in catalog:
        click.echo(f"<h2>{task_name}</h2>")
        default_model = catalog[task_name]["default"]
        for task_model in catalog[task_name]["models"]:
            task_class = catalog[task_name]["models"][task_model]["task"]
            is_default = (default_model is None and task_model is None) or task_model == default_model
            if task_model is None:
                model_label = "(default)"
            else:
                model_label = f"{task_model} (default)" if is_default else task_model
            model_file_urls = task_class.required_files(allow_optional=False)
            model_file_names = [get_file_name_from_url(file) for file in model_file_urls]
            model_file_sizes = [get_file_size_from_url(file) for file in model_file_urls]
            total_file_size = sum([size for size in model_file_sizes if size is not None])
            if model_file_urls:
                if len(model_file_urls) > 1:
                    model_files = "<ol>"
                    for name, url, size in zip(model_file_names, model_file_urls, model_file_sizes):
                         model_files += "<li><a href=\"{0}\" target=\"_blank\">{1}</a></li>".format(
                            url,
                            name if size is None else f"{name} ({human_size(size)})"
                         )
                    model_files += f"</ol><p><strong>Total Size</strong>: {human_size(total_file_size)}</p>"
                else:
                    model_files = f"<a href=\"{model_file_urls[0]}\" target=\"_blank\">{model_file_names[0]}</a>"
            else:
                model_files = "N/A"

            model_vram = None if not task_class.requires_gpu() else task_class.required_static_gpu_memory_gb()
            if model_vram is None:
                vram_label = "N/A"
            else:
                vram_label = f"{human_size(model_vram * 1000 * 1000 * 1000)}"

            if task_model is not None or len(catalog[task_name]["models"]) > 1:
                click.echo(f"<h3>{model_label}</h3>")

            click.echo(
                trim_html_whitespace(
                    tabulate.tabulate(
                        [
                            ["Name", task_class.get_display_name()],
                            ["Author", task_class.get_author_citation().replace("\n", "<br />")],
                            ["License", task_class.get_license_citation().replace("\n", "<br />")],
                            ["Files", model_files],
                            ["Minimum VRAM", vram_label],
                        ],
                        tablefmt="unsafehtml"
                    )
                )
            )

@main.command(name="tasks", short_help="Print installed task catalog.")
@context_options
def tasks(
    log_level: str="warning",
    add_import: List[str]=[]
) -> None:
    """
    Print local task catalog.
    """
    with get_command_context(log_level, add_import):
        from .tasks import Task
        available_tasks: Dict[str, List[str]] = {}
        unavailable_tasks: Dict[str, List[str]] = {}

        for task_name, model_name, task_class in Task.enumerate(available_only=False):
            model_display_name = "none" if model_name is None else model_name
            if task_class.default:
                model_display_name = f"{model_display_name}*"
            if task_class.is_available():
                if task_name not in available_tasks:
                    available_tasks[task_name] = []
                available_tasks[task_name].append(model_display_name)
            else:
                if task_name not in unavailable_tasks:
                    unavailable_tasks[task_name] = []
                unavailable_tasks[task_name].append(model_display_name)

        click.echo("Available tasks (* = default):")
        for task_name, model_names in available_tasks.items():
            click.echo(f"  {task_name}: {', '.join(model_names)}")
        if len(unavailable_tasks) > 0:
            click.echo("Unavailable tasks (* = default):")
            for task_name, model_names in unavailable_tasks.items():
                click.echo(f"  {task_name}: {', '.join(model_names)}")

@main.command(name="info", short_help="Print task details.")
@click.argument("task", type=str)
@click.argument("model", type=str, required=False)
@click.option("--model-dir", "-m", type=str, default=DEFAULT_MODEL_DIR, help="Model directory.", show_default=True)
@click.option("--optional/--no-optional", default=False, is_flag=True, show_default=True, help="Include optional dependencies.")
@context_options
def info(
    task: str,
    model: Optional[str]=None,
    model_dir: str=DEFAULT_MODEL_DIR,
    optional: bool=False,
    log_level: str="warning",
    add_import: List[str]=[]
) -> None:
    """
    Prints details for tasks that can be ran.
    """
    if ":" in task and model is None:
        task, _, model = task.partition(":")
    with get_command_context(log_level, add_import):
        from .tasks import Task
        from .util import (
            file_is_downloaded_to_dir,
            get_file_name_from_url,
            get_file_size_from_url,
            installed_package_matches_spec,
            green,
            yellow,
            cyan,
            red,
            blue,
            magenta,
            human_size
        )

        task_class = Task.get(
            task,
            model,
            available_only=False,
            model_dir=model_dir
        )
        if task_class is None:
            task_label = task
            if model is not None:
                task_label = f"{task}:{model}"
            click.echo(red(f"Task {task_label} not found."))
            return

        task_is_available = task_class.is_available(allow_optional=optional, model_dir=model_dir)
        task_files = task_class.required_files(allow_optional=optional)
        task_packages = task_class.combined_required_packages(allow_optional=optional)
        task_signature = task_class.introspect()
        task_author = task_class.get_author_citation()
        task_license = task_class.get_license_citation()

        task_uses_gpu = task_class.requires_gpu()
        task_precision = task_class.required_gpu_precision()
        task_required_memory_gb = task_class.required_static_memory_gb()
        task_required_gpu_memory_gb = task_class.required_static_gpu_memory_gb()

        if task_license:
            task_license_allowances = task_class.get_license_allowances()
        else:
            task_license_allowances = ""

        available_label = green("available") if task_is_available else red("unavailable")
        click.echo(f"{cyan(task_class.get_display_name())} ({task_class.get_key()}, {available_label})")
        if task_signature.get("short_description", None):
            click.echo(f"    {task_signature['short_description']}")
        if task_signature.get("long_description", None):
            click.echo(f"    {task_signature['long_description']}")
        click.echo("Hardware Requirements:")
        if task_uses_gpu:
            click.echo(f"    {yellow('GPU Required for Optimal Performance')}")
            if task_precision:
                click.echo(f"    {yellow('Floating Point Precision', False)}: {task_precision}")
        else:
            click.echo(f"    {green('No GPU Required')}")
        if task_required_memory_gb:
            num_bytes = task_required_memory_gb * 1024 * 1024 * 1024
            click.echo(f"    {blue('Minimum Memory (CPU RAM) Required')}: {human_size(num_bytes)}")
        if task_uses_gpu and task_required_gpu_memory_gb:
            num_bytes = task_required_gpu_memory_gb * 1024 * 1024 * 1024
            click.echo(f"    {blue('Minimum Memory (GPU VRAM) Required')}: {human_size(num_bytes)}")
        if task_author:
            click.echo("Author:")
            for i, line in enumerate(task_author.splitlines()):
                if i == 0:
                    click.echo(f"    {blue(line)}")
                else:
                    click.echo(f"    {line}")
        if task_license:
            click.echo("License:")
            click.echo(f"    {blue(task_license)}")
            if task_license_allowances:
                for line in task_license_allowances.splitlines():
                    click.echo(f"    {line}")
        if task_files:
            total_size = 0
            click.echo("Files:")
            for file in task_files:
                file_name = get_file_name_from_url(file)
                file_size = get_file_size_from_url(file)

                if file_is_downloaded_to_dir(file, model_dir):
                    downloaded_label = green("[downloaded]")
                else:
                    downloaded_label = red("[not downloaded]")

                if file_size is not None:
                    total_size += file_size
                    size_label = f" ({human_size(file_size)})"
                else:
                    size_label = ""

                click.echo(f"    {blue(file_name)}{size_label} {downloaded_label}")
            if total_size > 0:
                click.echo(f"    {cyan('Total File Size')}: {human_size(total_size)}")
        if task_packages:
            click.echo("Required packages:")
            for required_package, spec in task_packages.items():
                if installed_package_matches_spec(required_package, spec):
                    installed_label = green("[installed]")
                else:
                    installed_label = red("[not installed]")

                click.echo(f"    {blue(required_package)}{spec or ''} {installed_label}")

        click.echo("Signature:")
        for param_name, param_config in task_signature["parameters"].items():
            param_type = param_config["parameter_type"]
            if isinstance(param_type, str):
                param_type_name = param_type
            else:
                param_type_name = getattr(param_type, "__name__", str(param_type)) # type: ignore[arg-type]
            param_required = param_config.get("required", False)
            param_default = param_config.get("default", NOTSET)
            param_label = f"    {blue(param_name)}: {magenta(param_type_name)}"
            if param_required:
                param_label += ", " + yellow("required")
            if param_default is not NOTSET:
                param_label += f", default: {param_default}"
            click.echo(param_label)
            if param_config.get("description", None):
                click.echo(f"        {param_config['description']}")

        if task_signature.get("return_type", None):
            if isinstance(task_signature["return_type"], str):
                return_type_name = task_signature["return_type"]
            else:
                return_type_name = getattr(task_signature["return_type"], "__name__", str(task_signature["return_type"]))
            click.echo("Returns:")
            click.echo(f"    {magenta(return_type_name)}")

@main.command(name="install", short_help="Installs pacakages and downloads files for a task.")
@click.argument("tasks", type=str, nargs=-1)
@click.option("--model-dir", "-m", type=str, default=DEFAULT_MODEL_DIR, help="Model directory.", show_default=True)
@click.option("--max-workers", "-w", type=int, default=4, help="Maximum number of workers for downloads.", show_default=True)
@click.option("--reinstall/--no-reinstall", default=False, is_flag=True, show_default=True, help="Reinstall packages.")
@click.option("--files/--no-files", default=True, is_flag=True, show_default=True, help="Download files.")
@click.option("--packages/--no-packages", default=True, is_flag=True, show_default=True, help="Install packages.")
@click.option("--optional/--no-optional", default=False, is_flag=True, show_default=True, help="Include optional dependencies.")
@context_options
def install(
    tasks: List[str]=[],
    files: bool=True,
    packages: bool=True,
    reinstall: bool=False,
    model_dir: str=DEFAULT_MODEL_DIR,
    max_workers: int=4,
    optional: bool=False,
    log_level: str="warning",
    add_import: List[str]=[]
) -> None:
    """
    Installs packages and downloads files for a task.
    """
    with get_command_context(log_level, add_import):
        from .tasks import Task
        from .util import (
            combine_package_specifications,
            install_packages,
            get_file_name_from_url,
            check_download_files_to_dir,
            red
        )

        target_tasks: List[Type[Task]] = []

        for task_name, model_name, task_class in Task.enumerate(available_only=False):
            if not tasks:
                target_tasks.append(task_class)
                continue
            for passed_task in tasks:
                passed_task_parts = passed_task.split(":")
                if len(passed_task_parts) == 1:
                    passed_task_name = passed_task_parts[0]
                    passed_task_model = None
                else:
                    passed_task_name, passed_task_model = passed_task_parts

                if task_name == passed_task_name:
                    if model_name == passed_task_model or passed_task_model is None:
                        target_tasks.append(task_class)
                        continue

        if not target_tasks:
            click.echo(red("No tasks could be found with the provided arguments."))
            return

        pending_downloads: List[str] = []
        pending_packages: List[Dict[str, Optional[str]]] = []

        for task_class in target_tasks:
            if files:
                pending_downloads.extend(
                    task_class.get_pending_downloads(
                        model_dir=model_dir,
                        allow_optional=optional
                    )
                 )
            if packages or reinstall:
                if reinstall:
                    pending_packages.append(
                        task_class.combined_required_packages(
                            allow_optional=optional
                        )
                    )
                else:
                    pending_packages.append(
                        task_class.get_pending_packages(
                            allow_optional=optional
                        )
                    )

        pending_downloads = list(set(pending_downloads)) # remove duplicates
        pending_package_spec = combine_package_specifications(*pending_packages)
        num_pending_downloads = len(pending_downloads)
        num_pending_packages = len(pending_package_spec)

        if num_pending_downloads == 0 and num_pending_packages == 0:
            click.echo("Nothing to install.")
            return

        if num_pending_downloads > 0 and files:
            click.echo(f"Downloading {num_pending_downloads} file(s).")
            try:
                from tqdm import tqdm
                progress_bars = [
                    tqdm(
                        desc=get_file_name_from_url(url),
                        total=1,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        mininterval=1.0
                    )
                    for url in pending_downloads
                ]
                progress_bar_update_times = [perf_counter()] * num_pending_downloads

                def progress_callback(
                    file_index: int,
                    files_total: int,
                    bytes_downloaded: int,
                    bytes_total: int
                ) -> None:
                    """
                    progress callback for updating progress bars.
                    """
                    if progress_bars[file_index].total != bytes_total:
                        progress_bars[file_index].reset(total=bytes_total)
                    progress_time = perf_counter()
                    if progress_time - progress_bar_update_times[file_index] > 1.0 or bytes_downloaded >= bytes_total:
                        progress_bars[file_index].n = bytes_downloaded
                        progress_bars[file_index].refresh()
                        progress_bar_update_times[file_index] = progress_time

            except ImportError:
                progress_callback = None # type: ignore[assignment]

            check_download_files_to_dir(
                pending_downloads,
                model_dir,
                max_workers=max_workers,
                progress_callback=progress_callback
            )

        if num_pending_packages > 0 and (packages or reinstall):
            click.echo(f"Installing {num_pending_packages} package(s).")
            install_packages(pending_package_spec, reinstall) # Uses pip

@main.command(name="echo", short_help="Runs an echo server for testing.")
@click.argument("address", type=str, required=False, default=DEFAULT_ADDRESS)
@context_options
def echo(
    address: str=DEFAULT_ADDRESS,
    log_level: str="warning",
    add_import: List[str]=[]
) -> None:
    """
    Runs an echo server for testing.
    """
    with get_command_context(log_level, add_import):
        from .server import Server
        server = Server()
        server.address = address
        loop = asyncio.new_event_loop()
        # Run server
        server_task = loop.create_task(server.run())
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.run_until_complete(server.assert_connectivity())
        # Wait forever
        click.echo(f"Echo server running at {address}.")
        try:
            while True:
                loop.run_until_complete(asyncio.sleep(0.1))
        except KeyboardInterrupt:
            pass
        # Graceful exit
        loop.run_until_complete(server.exit())
        loop.run_until_complete(asyncio.sleep(0.1))
        tasks = asyncio.all_tasks(loop)
        try:
            for task in tasks:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        except:
            pass
        loop.close()

@main.command(name="overseer", short_help="Runs an overseer (cluster entrypoint and node manager).")
@click.argument("address", type=str, required=False, default=DEFAULT_ADDRESS)
@click.option("--dispatcher", "-d", multiple=True, type=str, help="Dispatcher address to register after starting.", show_default=True)
@click.option("--config", "-c", type=click.Path(exists=True), default=None, help="Configuration file to use.", show_default=True)
@click.option("--local", "-l", type=bool, default=False, help="Additionally run a local dispatcher while running the overseer.", show_default=True, is_flag=True)
@click.option("--dispatcher-config", "-dc", type=click.Path(exists=True), default=None, help="Dispatcher configuration file to use. Overrides other dispatcher-related configuration.", show_default=True)
@click.option("--allow", "-a", type=str, multiple=True, help="Allow list for all tcp connections.", show_default=True)
@click.option("--allow-control", "-ac", type=str, multiple=True, help="Allow list for dispatcher connections.", show_default=True)
@click.option("--max-workers", "-w", type=int, default=1, help="Maximum number of workers for executors when using local mode. When --local/-l is not passed, has no effect.", show_default=True)
@click.option("--queue-size", "-qs", type=int, default=1, help="Maximum queue size for executors when using local mode. When --local/-l is not passed, has no effect.", show_default=True)
@click.option("--save-dir", "-s", type=str, default=None, help="Directory to save files to when using local mode. When --local/-l is not passed, has no effect.", show_default=True)
@click.option("--certfile", "-cf", type=click.Path(exists=True), default=None, help="SSL certificate file when using WSS.", show_default=True)
@click.option("--keyfile", "-kf", type=click.Path(exists=True), default=None, help="SSL key file when using WSS.", show_default=True)
@click.option("--cafile", "-caf", type=click.Path(exists=True), default=None, help="SSL CA file when using WSS.", show_default=True)
@click.option("--control-encryption-key", "-cek", type=str, default=None, help="Encryption key for control messages.", show_default=True)
@click.option("--pidfile", "-p", type=click.Path(), default=None, help="PID file to write to.", show_default=True)
@click.option("--exclusive", "-e", type=bool, default=False, show_default=True, is_flag=True, help="Exclusively run one overseer (requires --pidfile).")
@context_options
def overseer(
    address: str=DEFAULT_ADDRESS,
    dispatcher: List[str]=[],
    config: Optional[str]=None,
    local: bool=False,
    dispatcher_config: Optional[str]=None,
    allow: List[str]=[],
    allow_control: List[str]=[],
    max_workers: int=1,
    queue_size: int=1,
    save_dir: Optional[str]=None,
    certfile: Optional[str]=None,
    keyfile: Optional[str]=None,
    cafile: Optional[str]=None,
    control_encryption_key: Optional[str]=None,
    pidfile: Optional[str]=None,
    exclusive: bool=False,
    log_level: str="warning",
    add_import: List[str]=[]
) -> None:
    """
    Runs an overseer (cluster entrypoint and node manager).

    Additionally runs a local dispatcher while running the overseer if --local/-l is passed.
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

    with get_command_context(log_level, add_import):
        from .server import Overseer, Dispatcher # Create overseer
        server = Overseer(config)
        if config is None or address != DEFAULT_ADDRESS:
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

        # Run overseer
        loop = asyncio.new_event_loop()
        server_task = loop.create_task(server.run())
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.run_until_complete(server.assert_connectivity())

        # Optionally run local dispatcher
        local_server: Optional[Dispatcher] = None
        local_task: Optional[asyncio.Task[Any]] = None
        if local:
            local_server = Dispatcher(dispatcher_config)
            if not dispatcher_config:
                local_server.protocol = "memory"
                local_server.port = 0
                local_server.max_workers = max_workers
                local_server.executor_queue_size = queue_size
            if save_dir is not None:
                local_server.save_dir = save_dir
            local_task = loop.create_task(local_server.run())
            loop.run_until_complete(asyncio.sleep(0.1))
            loop.run_until_complete(local_server.assert_connectivity())
            loop.run_until_complete(local_server.register_overseer(address))

        # Register dispatchers
        for dispatcher_address in dispatcher:
            server.register_dispatcher(dispatcher_address)

        # Wait forever
        click.echo(f"Overseer running at {server.address}.")
        if local and local_server is not None:
            click.echo(f"Local dispatcher running at {local_server.address}.")

        try:
            while True:
                loop.run_until_complete(asyncio.sleep(0.1))
        except KeyboardInterrupt:
            pass

        # Graceful exit
        loop.run_until_complete(server.exit())
        if local and local_server is not None:
            loop.run_until_complete(local_server.exit())

        # Sleep and then cancel any remaining tasks
        loop.run_until_complete(asyncio.sleep(0.1))
        tasks = asyncio.all_tasks(loop)
        try:
            for task in tasks:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        except:
            pass
        loop.close()
        if pidfile:
            try:
                os.remove(pidfile)
            except:
                pass

@main.command(name="dispatcher", short_help="Runs a dispatcher.")
@click.argument("address", type=str, required=False, default=DEFAULT_DISPATCHER_ADDRESS)
@click.option("--overseer", type=str, help="Overseer address to register with.", multiple=True, show_default=True)
@click.option("--config", "-c", type=click.Path(exists=True), default=None, help="Configuration file to use.", show_default=True)
@click.option("--allow", "-a", type=str, multiple=True, help="Allow list for all tcp connections.", show_default=True)
@click.option("--allow-control", "-ac", type=str, multiple=True, help="Allow list for dispatcher connections.", show_default=True)
@click.option("--max-workers", "-w", type=int, default=None, help="Maximum number of workers for executors.", show_default=True)
@click.option("--queue-size", "-qs", type=int, default=None, help="Maximum queue size for executors.", show_default=True)
@click.option("--certfile", "-cf", type=click.Path(exists=True), default=None, help="SSL certificate file when using WSS.", show_default=True)
@click.option("--keyfile", "-kf", type=click.Path(exists=True), default=None, help="SSL key file when using WSS.", show_default=True)
@click.option("--cafile", "-caf", type=click.Path(exists=True), default=None, help="SSL CA file when using WSS.", show_default=True)
@click.option("--control-encryption-key", "-cek", type=str, default=None, help="Encryption key for control messages.", show_default=True)
@click.option("--pidfile", "-p", type=click.Path(), default=None, help="PID file to write to.", show_default=True)
@click.option("--exclusive", "-e", type=bool, default=False, show_default=True, is_flag=True, help="Exclusively run one overseer (requires --pidfile).")
@context_options
def dispatcher(
    address: str=DEFAULT_DISPATCHER_ADDRESS,
    overseer: List[str]=[],
    config: Optional[str]=None,
    allow: List[str]=[],
    allow_control: List[str]=[],
    max_workers: Optional[int]=None,
    queue_size: Optional[int]=None,
    certfile: Optional[str]=None,
    keyfile: Optional[str]=None,
    cafile: Optional[str]=None,
    control_encryption_key: Optional[str]=None,
    pidfile: Optional[str]=None,
    exclusive: bool=False,
    log_level: str="warning",
    add_import: List[str]=[]
) -> None:
    """
    Runs a dispatcher.
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

    with get_command_context(log_level, add_import):
        from .server import Dispatcher
        server = Dispatcher(config)
        if config is None or address != DEFAULT_DISPATCHER_ADDRESS:
            server.address = address
        if allow:
            server.allow_list = list(allow)
        if allow_control:
            server.control_list = list(allow_control)
        if max_workers is not None:
            server.max_workers = max_workers
        if queue_size is not None:
            server.executor_queue_size = queue_size
        if certfile is not None:
            server.certfile = certfile
        if keyfile is not None:
            server.keyfile = keyfile
        if cafile is not None:
            server.cafile = cafile
        if control_encryption_key is not None:
            server.use_control_encryption = True
            server.control_encryption_key = control_encryption_key # type: ignore[assignment]

        loop = asyncio.new_event_loop()
        # Run server
        server_task = loop.create_task(server.run())
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.run_until_complete(server.assert_connectivity())

        # Register dispatcher with overseers
        for overseer_address in overseer:
            loop.run_until_complete(server.register_overseer(overseer_address))

        # Wait forever
        click.echo(f"Dispatcher running at {server.address}.")
        try:
            while True:
                loop.run_until_complete(asyncio.sleep(0.1))
        except KeyboardInterrupt:
            pass
        # Graceful exit
        for overseer_address in overseer:
            loop.run_until_complete(server.unregister_overseer(overseer_address))
        loop.run_until_complete(server.exit())

        # Sleep and then cancel any remaining tasks
        loop.run_until_complete(asyncio.sleep(0.1))
        tasks = asyncio.all_tasks(loop)
        try:
            for task in tasks:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        except:
            pass
        loop.close()
        if pidfile:
            try:
                os.remove(pidfile)
            except:
                pass

@main.command(name="chat", short_help="Chat with an AI model.")
@click.argument("model", type=str, required=False, default=None)
@click.option("--model-dir", "-m", type=str, default=DEFAULT_MODEL_DIR, help="Model directory.", show_default=True)
@click.option("--forgetful", "-f", type=bool, default=False, help="Forget previous context.", show_default=True, is_flag=True)
@click.option("--stream", "-st", type=bool, default=False, help="Stream output.", show_default=True, is_flag=True)
@click.option("--role", "-r", type=str, default=None, help="Role to chat as.", show_default=True)
@click.option("--seed", "-s", type=int, default=None, help="Seed for randomness.", show_default=True)
@click.option("--use-tools", "-t", is_flag=True, help="Use tools for chat.")
@click.option("--max-tokens", "-mt", type=int, default=None, help="Maximum tokens to generate.", show_default=True)
@click.option("--context-length", "-cl", type=int, default=None, help="Context length. Default uses the full context as configured in the model", show_default=True)
@context_options
def chat(
    model: Optional[str]=None,
    model_dir: str=DEFAULT_MODEL_DIR,
    forgetful: bool=False,
    stream: bool=False,
    role: Optional[str]=None,
    seed: Optional[int]=None,
    use_tools: bool=False,
    max_tokens: Optional[int]=None,
    context_length: Optional[int]=None,
    log_level: str="warning",
    add_import: List[str]=[]
) -> None:
    """
    Chat with an AI model.
    """
    with get_command_context(log_level, add_import):
        from .tasks import TaskQueue
        from .util import (
            magenta,
            green,
            cyan,
            red
        )
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        queue = TaskQueue.get(
            "text-generation",
            model=model,
            task_config={
                "context_length": context_length
            }
        )
        conversation: List[str] = []
        loop.run_until_complete(queue._wait_for_task())

        if seed is None:
            seed = randint(0x10000000, 0xFFFFFFFF)

        system = magenta("[system]")
        assistant = cyan("[assistant]")
        user = green("[user]")
        brk = "\033[K"

        click.echo(f"{system} Model set to {queue._task.model}.{brk}")
        click.echo(f"{system} Seed set to {seed}.{brk}")

        try:
            while True:
                prompt = input(f"{user} {brk}")
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
                    loop.run_until_complete(asyncio.sleep(0.05))
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
        except (EOFError, KeyboardInterrupt):
            del queue
            click.echo(f"\r{system} Goodbye!{brk}")

@main.command(name="invoke", short_help="Invoke a task on either a remote or local cluster.", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("task", type=str)
@click.argument("model", type=str, required=False)
@click.option("--output-format", "-of", type=str, default=None, help="Output format, when output is media. Valid options depend on the media type. Defaults are png for images, mp4 for videos, and wav for audio.", show_default=True)
@click.option("--output-dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory, when output is media.", show_default=True)
@click.option("--model-dir", "-m", type=str, default=DEFAULT_MODEL_DIR, help="Model directory.", show_default=True)
@click.option("--model-offload/--no-model-offload", default=False, is_flag=True, show_default=True, help="Offload models to CPU after use in supported pipelines.")
@click.option("--sequential-offload/--no-sequential-offload", default=False, is_flag=True, show_default=True, help="Offload layers to CPU after use in supported pipelines.")
@click.option("--encode-tiling/--no-encode-tiling", default=False, is_flag=True, show_default=True, help="Enable tiled encoding in supported pipelines.")
@click.option("--encode-slicing/--no-encode-slicing", default=False, is_flag=True, show_default=True, help="Enable sliced encoding in supported pipelines.")
@click.option("--context-length", "-cl", default=None, type=int, help="Context length for supported pipelines.", show_default=True)
@click.option("--open-output/--no-open-output", default=True, is_flag=True, show_default=True, help="Open an output file after completion. Only applies to tasks that produce output files.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all output except for the result.")
@click.option("--json", "-j", is_flag=True, help="Output result as JSON.")
@context_options
def invoke(
    task: str,
    model: Optional[str]=None,
    output_format: Optional[str]=None,
    output_dir: str=DEFAULT_OUTPUT_DIR,
    model_dir: str=DEFAULT_MODEL_DIR,
    model_offload: bool=False,
    sequential_offload: bool=False,
    encode_tiling: bool=False,
    encode_slicing: bool=False,
    context_length: Optional[int]=None,
    open_output: bool=True,
    quiet: bool=False,
    json: bool=False,
    log_level: str="warning",
    add_import: List[str]=[]
) -> None:
    """
    Invoke a task on either a remote or local cluster.
    """
    num_args = len(sys.argv)
    skip = False
    args: Dict[str, Any] = {}

    if model is not None and model.startswith("--"):
        model = None

    if ":" in task and model is None:
        task, _, model = task.partition(":")

    for i, arg in enumerate(sys.argv):
        if skip:
            skip = False
            continue

        if arg.startswith("--"):
            flag_parts = arg.split("=")
            flag = flag_parts[0][2:].replace("-", "_")
            if flag in [
                "debug", "info", "warning",
                "error", "critical", "output_format",
                "output_dir", "model_dir", "model",
                "model_offload", "sequential_offload",
                "encode_tiling", "encode_slicing",
                "context_length", "open_output", "no_open_output",
                "quiet", "json", "o", "f", "of",
                "q", "j", "m", "cl",
            ]:
                continue

            value: Any = True
            if len(flag_parts) > 1:
                value = flag_parts[1]
            elif i + 1 < num_args:
                if not sys.argv[i + 1].startswith("-"):
                    value = sys.argv[i + 1]
                    skip = True

            args[flag] = value

    with get_command_context(log_level, add_import):
        from .tasks import Task
        from .util import (
            validate_parameters,
            time_counter,
            human_duration,
            open_file,
            yellow,
            green,
            cyan,
            red
        )
        task_class = Task.get(
            task,
            model,
            available_only=False,
            model_dir=model_dir
        )

        if task_class is None:
            task_label = task
            if model is not None:
                task_label = f"{task} ({model})"
            click.echo(red(f"Task {task_label} not found."))
            return

        task_is_available = task_class.is_available(model_dir=model_dir)
        if not task_is_available:
            command = f"taproot install {task}"
            if model is not None:
                command += f" {model}"
            click.echo(red(f"Task {task_class.get_key()} is not available, run '{command}' to install dependencies and download files."))
            return

        task_signature = task_class.introspect()
        task_parameters = task_signature["parameters"]
        invoke_args = validate_parameters(
            task_parameters,
            args,
            include_defaults=True,
            raise_on_missing=True,
            raise_on_invalid=True,
            raise_on_extra=False
        )

        # Invocation args are good, instantiate, load and invoke
        if not quiet: click.echo(yellow("Loading task."))
        task_instance = task_class()
        task_instance.save_dir = output_dir
        task_instance.model_dir = model_dir
        task_instance.enable_model_offload = model_offload
        task_instance.enable_sequential_offload = sequential_offload
        task_instance.enable_encode_tiling = encode_tiling
        task_instance.enable_encode_slicing = encode_slicing
        task_instance.context_length = context_length
        task_instance.use_tqdm = True

        with time_counter() as timer:
            task_instance.load()

        if not quiet: click.echo(f"Task loaded in {cyan(human_duration(float(timer)))}.")

        if "output_format" in task_parameters:
            invoke_args["output_format"] = output_format
        if "output_upload" in task_parameters:
            invoke_args["output_upload"] = True

        if not quiet: click.echo(yellow("Invoking task."))
        with time_counter() as timer:
            result = task_instance(**invoke_args)

        if not quiet: click.echo(f"Task invoked in {cyan(human_duration(float(timer)))}. Result:")
        if json:
            import json as pyjson
            click.echo(pyjson.dumps(result))
        else:
            click.echo(green(result))

        with time_counter() as timer:
            task_instance.unload()

        if not quiet: click.echo(f"Task unloaded in {cyan(human_duration(float(timer)))}.")

        if open_output and isinstance(result, str) and os.path.exists(result):
            open_file(result)

try:
    main()
    sys.exit(0)
except Exception as ex:
    sys.stderr.write(f"{ex}\r\n")
    sys.stderr.write(traceback.format_exc())
    sys.stderr.flush()
    sys.exit(5)
