from __future__ import annotations

import os
import sys
import click
import traceback
import functools

from time import perf_counter

from typing import Dict, List, Optional, Type, Any, Set, TYPE_CHECKING

from .version import version
from .constants import *
from .util import (
    context_options,
    server_options,
    get_pidfile_context,
    get_command_context,
    get_server,
    get_server_runner,
    async_chat_loop,
    trim_html_whitespace,
    AsyncRunner,
)

if TYPE_CHECKING:
    from .server import Server

@click.group(name="taproot")
@click.version_option(version=str(version), message="%(version)s")
def main() -> None:
    """
    Taproot command-line tools.
    """
    pass

@main.command(name="machine-capability", short_help="Print machine capability.")
@context_options(include_quiet=False)
def machine_capability(
    log_level: str=DEFAULT_LOG_LEVEL,
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
@context_options(include_quiet=False, include_model_dir=False)
def catalog(
    log_level: str=DEFAULT_LOG_LEVEL,
    add_import: List[str]=[]
) -> None:
    """
    Prints complete catalog of available tasks in tabular format.
    """
    with get_command_context(log_level, add_import):
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
@context_options(include_quiet=False)
def tasks(
    model_dir: str=DEFAULT_MODEL_DIR,
    log_level: str=DEFAULT_LOG_LEVEL,
    add_import: List[str]=[]
) -> None:
    """
    Print local task catalog.
    """
    with get_command_context(log_level, add_import):
        from .tasks import Task
        available_tasks: Dict[str, List[str]] = {}
        unavailable_tasks: Dict[str, List[str]] = {}

        for task_name, model_name, task_class in Task.enumerate(
            model_dir=model_dir,
            available_only=False
        ):
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
@click.option("--optional/--no-optional", default=False, is_flag=True, show_default=True, help="Include optional dependencies.")
@context_options(include_quiet=False)
def info(
    task: str,
    model: Optional[str]=None,
    optional: bool=False,
    model_dir: str=DEFAULT_MODEL_DIR,
    log_level: str=DEFAULT_LOG_LEVEL,
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
            get_pip_package_name,
            installed_package_matches_spec,
            required_library_is_available,
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
        task_libraries = task_class.required_libraries(allow_optional=optional)
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
        if task_libraries:
            click.echo("Required libraries:")
            for library in task_libraries:
                if required_library_is_available(library):
                    available_label = green("[available]")
                else:
                    available_label = red("[not available]")
                click.echo(f"    {blue(library['name'])} {available_label}")
        if task_files:
            total_size = 0
            click.echo("Required files:")
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

                click.echo(f"    {blue(get_pip_package_name(required_package))}{spec or ''} {installed_label}")

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

@main.command(name="packages", short_help="Print combined required packages for one or more tasks.")
@click.argument("tasks", type=str, nargs=-1)
@click.option("--optional/--no-optional", default=False, is_flag=True, show_default=True, help="Include optional dependencies.")
@context_options(include_save_dir=False, include_model_dir=False, include_quiet=False)
def packages(
    tasks: List[str]=[],
    optional: bool=False,
    log_level: str=DEFAULT_LOG_LEVEL,
    add_import: List[str]=[],
) -> None:
    """
    Prints combined required packages for one or more tasks, one per line.

    Does not print versions with attached precompiled libraries.
    """
    with get_command_context(log_level, add_import):
        from .tasks import Task
        from .util import (
            assert_required_library_installed,
            check_download_files_to_dir,
            combine_package_specifications,
            get_file_name_from_url,
            get_pip_package_name,
            install_packages,
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

        packages: List[Dict[str, Optional[str]]] = []

        for task_class in target_tasks:
            packages.append(
                task_class.combined_required_packages(
                    allow_optional=optional
                )
            )

        combined_packages = combine_package_specifications(*packages)

        if not combined_packages:
            return

        click.echo(
            "\n".join([
                f"{get_pip_package_name(package)}{spec or ''}"
                for package, spec in combined_packages.items()
            ])
        )

@main.command(name="files", short_help="Print combined required files for one or more tasks.")
@click.argument("tasks", type=str, nargs=-1)
@click.option("--optional/--no-optional", default=False, is_flag=True, show_default=True, help="Include optional dependencies.")
@context_options(include_save_dir=False, include_model_dir=False, include_quiet=False)
def files(
    tasks: List[str]=[],
    optional: bool=False,
    log_level: str=DEFAULT_LOG_LEVEL,
    add_import: List[str]=[],
) -> None:
    """
    Prints combined required files for one or more tasks, one per line.
    """
    with get_command_context(log_level, add_import):
        from .tasks import Task
        from .util import (
            assert_required_library_installed,
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

        files: Set[str] = set()

        for task_class in target_tasks:
            files.update(task_class.required_files(allow_optional=optional))

        click.echo("\n".join(files))

@main.command(name="install", short_help="Installs pacakages and downloads files for a task.")
@click.argument("tasks", type=str, nargs=-1)
@click.option("--max-workers", "-w", type=int, default=4, help="Maximum number of workers for downloads.", show_default=True)
@click.option("--reinstall/--no-reinstall", default=False, is_flag=True, show_default=True, help="Reinstall packages.")
@click.option("--files/--no-files", default=True, is_flag=True, show_default=True, help="Download files.")
@click.option("--packages/--no-packages", default=True, is_flag=True, show_default=True, help="Install packages.")
@click.option("--optional/--no-optional", default=False, is_flag=True, show_default=True, help="Include optional dependencies.")
@context_options(include_save_dir=False)
def install(
    tasks: List[str]=[],
    files: bool=True,
    packages: bool=True,
    reinstall: bool=False,
    max_workers: int=4,
    optional: bool=False,
    model_dir: str=DEFAULT_MODEL_DIR,
    log_level: str=DEFAULT_LOG_LEVEL,
    add_import: List[str]=[],
    quiet: bool=False
) -> None:
    """
    Installs packages and downloads files for a task.
    """
    with get_command_context(log_level, add_import, quiet):
        from .tasks import Task
        from .util import (
            assert_required_library_installed,
            combine_package_specifications,
            install_packages,
            get_file_name_from_url,
            check_download_files_to_dir,
            red
        )

        target_tasks: List[Type[Task]] = []

        for task_name, model_name, task_class in Task.enumerate(available_only=False, model_dir=model_dir):
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
            # Check for libraries first, we don't install these automatically so we need to stop here
            # if they aren't available. This will print an appropriate install command if one is known.
            for required_library in task_class.required_libraries(allow_optional=optional):
                assert_required_library_installed(required_library)

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
            if not quiet:
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
            else:
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
@server_options()
@context_options(include_model_dir=False, include_save_dir=False)
def echo(
    address: str=DEFAULT_ADDRESS,
    config: Optional[str]=None,
    allow: List[str]=[],
    reject: List[str]=[],
    allow_control: List[str]=[],
    certfile: Optional[str]=None,
    keyfile: Optional[str]=None,
    cafile: Optional[str]=None,
    control_encryption_key: Optional[str]=None,
    pidfile: Optional[str]=None,
    exclusive: bool=False,
    log_level: str=DEFAULT_LOG_LEVEL,
    add_import: List[str]=[],
    quiet: bool=False
) -> None:
    """
    Runs an echo server for testing.
    """
    with get_pidfile_context(pidfile, exclusive):
        with get_command_context(log_level, add_import, quiet):
            from .server import ConfigServer
            server = get_server(
                ConfigServer,
                address=address,
                config=config,
                allow=allow,
                reject=reject,
                allow_control=allow_control,
                certfile=certfile,
                keyfile=keyfile,
                cafile=cafile,
                control_encryption_key=control_encryption_key
            )
            get_server_runner(server).run(debug=log_level.upper()=="DEBUG")

@main.command(name="overseer", short_help="Runs an overseer (cluster entrypoint and node manager).")
@click.option("--local", "-l", type=bool, default=False, help="Additionally run a local dispatcher while running the overseer.", show_default=True, is_flag=True)
@click.option("--local-address", "-la", type=str, default=DEFAULT_LOCAL_DISPATCHER_ADDRESS, help="Local dispatcher address to use.", show_default=True)
@click.option("--local-config", "-lc", type=click.Path(exists=True), default=None, help="Local dispatcher configuration file to use. Overrides other dispatcher-related configuration.", show_default=True)
@click.option("--local-max-workers", "-lw", type=int, default=1, help="Maximum number of workers for executors when using local mode. When --local/-l is not passed, has no effect.", show_default=True)
@click.option("--local-queue-size", "-lq", type=int, default=1, help="Maximum queue size for executors when using local mode. When --local/-l is not passed, has no effect.", show_default=True)
@click.option("--dispatcher", "-d", multiple=True, type=str, help="Dispatcher address(es) to register after starting.", show_default=True)
@context_options()
@server_options()
def overseer(
    local: bool=False,
    local_address: str=DEFAULT_LOCAL_DISPATCHER_ADDRESS,
    local_config: Optional[str]=None,
    local_max_workers: int=1,
    local_queue_size: int=1,
    dispatcher: List[str]=[],
    address: str=DEFAULT_ADDRESS,
    config: Optional[str]=None,
    allow: List[str]=[],
    reject: List[str]=[],
    allow_control: List[str]=[],
    model_dir: str=DEFAULT_MODEL_DIR,
    save_dir: Optional[str]=None,
    certfile: Optional[str]=None,
    keyfile: Optional[str]=None,
    cafile: Optional[str]=None,
    control_encryption_key: Optional[str]=None,
    pidfile: Optional[str]=None,
    exclusive: bool=False,
    log_level: str=DEFAULT_LOG_LEVEL,
    add_import: List[str]=[],
    quiet: bool=False
) -> None:
    """
    Runs an overseer (cluster entrypoint and node manager).

    Additionally runs a local dispatcher while running the overseer if --local/-l is passed.
    """
    with get_pidfile_context(pidfile, exclusive):
        with get_command_context(log_level, add_import):
            from .server import Overseer, Dispatcher
            # Create overseer
            server = get_server(
                Overseer,
                address=address,
                config=config,
                allow=allow,
                reject=reject,
                allow_control=allow_control,
                certfile=certfile,
                keyfile=keyfile,
                cafile=cafile,
                control_encryption_key=control_encryption_key
            )
            local_server: Optional[Server] = None
            servers: List[Server] = [server]

            # Optionally run local dispatcher
            if local:
                local_server = get_server(
                    Dispatcher,
                    address=local_address,
                    default_address=DEFAULT_LOCAL_DISPATCHER_ADDRESS,
                    config=local_config,
                    allow=allow,
                    reject=reject,
                    allow_control=allow_control,
                    certfile=certfile,
                    keyfile=keyfile,
                    cafile=cafile,
                    control_encryption_key=control_encryption_key
                )
                if not local_config:
                    if not local_address:
                        local_server.protocol = "memory"
                        local_server.port = 0
                    else:
                        local_server.address = local_address
                    local_server.max_workers = local_max_workers
                    local_server.executor_queue_size = local_queue_size

                if save_dir is not None:
                    local_server.save_dir = save_dir

                if model_dir is not None:
                    local_server.model_dir = model_dir

                servers.append(local_server)

            runner = get_server_runner(*servers)

            async def after_start() -> None:
                """
                Register dispatchers after starting.

                We don't need to unregister them as the default exit handlers
                take care of unregistering all dispatchers.
                """
                # Register remote dispatchers
                for dispatcher_address in dispatcher:
                    server.register_dispatcher(dispatcher_address)

                # Register local dispatcher
                if local and local_server is not None:
                    server.register_dispatcher(local_server.address)

            runner.after_start(after_start)
            runner.run(debug=log_level.upper()=="DEBUG")

@main.command(name="dispatcher", short_help="Runs a dispatcher.")
@click.option("--overseer", type=str, help="Overseer address to register with.", multiple=True, show_default=True)
@click.option("--max-workers", "-w", type=int, default=None, help="Maximum number of workers for executors.", show_default=True)
@click.option("--queue-size", "-qs", type=int, default=None, help="Maximum queue size for executors.", show_default=True)
@server_options(default_address=DEFAULT_DISPATCHER_ADDRESS)
@context_options()
def dispatcher(
    overseer: List[str]=[],
    max_workers: Optional[int]=None,
    queue_size: Optional[int]=None,
    address: str=DEFAULT_DISPATCHER_ADDRESS,
    config: Optional[str]=None,
    model_dir: str=DEFAULT_MODEL_DIR,
    save_dir: Optional[str]=None,
    allow: List[str]=[],
    reject: List[str]=[],
    allow_control: List[str]=[],
    certfile: Optional[str]=None,
    keyfile: Optional[str]=None,
    cafile: Optional[str]=None,
    control_encryption_key: Optional[str]=None,
    pidfile: Optional[str]=None,
    exclusive: bool=False,
    log_level: str=DEFAULT_LOG_LEVEL,
    add_import: List[str]=[],
    quiet: bool=False
) -> None:
    """
    Runs a dispatcher.
    """
    with get_pidfile_context(pidfile, exclusive):
        with get_command_context(log_level, add_import, quiet):
            from .server import Dispatcher
            server = get_server(
                Dispatcher,
                address=address,
                config=config,
                allow=allow,
                reject=reject,
                allow_control=allow_control,
                certfile=certfile,
                keyfile=keyfile,
                cafile=cafile,
                control_encryption_key=control_encryption_key
            )

            if max_workers is not None:
                server.max_workers = max_workers
            if queue_size is not None:
                server.executor_queue_size = queue_size
            if save_dir is not None:
                server.save_dir = save_dir
            if model_dir is not None:
                server.model_dir = model_dir

            get_server_runner(server).run(debug=log_level.upper()=="DEBUG")

@main.command(name="chat", short_help="Chat with an AI model.")
@click.argument("model", type=str, required=False, default=None)
@click.option("--forgetful", "-f", type=bool, default=False, help="Forget previous context.", show_default=True, is_flag=True)
@click.option("--stream", "-st", type=bool, default=False, help="Stream output.", show_default=True, is_flag=True)
@click.option("--role", "-r", type=str, default=None, help="Role to chat as.", show_default=True)
@click.option("--seed", "-s", type=int, default=None, help="Seed for randomness.", show_default=True)
@click.option("--use-tools", "-t", is_flag=True, help="Use tools for chat.")
@click.option("--max-tokens", "-mt", type=int, default=None, help="Maximum tokens to generate.", show_default=True)
@click.option("--context-length", "-cl", type=int, default=None, help="Context length. Default uses the full context as configured in the model", show_default=True)
@context_options(include_save_dir=False, include_quiet=False)
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
    log_level: str=DEFAULT_LOG_LEVEL,
    add_import: List[str]=[]
) -> None:
    """
    Chat with an AI model.
    """
    chat_loop = functools.partial(
        async_chat_loop,
        model=model,
        model_dir=model_dir,
        forgetful=forgetful,
        stream=stream,
        role=role,
        seed=seed,
        use_tools=use_tools,
        max_tokens=max_tokens,
        context_length=context_length,
        log_level=log_level,
        add_import=add_import
    )
    AsyncRunner(chat_loop).run(debug=log_level.upper()=="DEBUG")

@main.command(name="invoke", short_help="Invoke a task on either a remote or local cluster.", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("task", type=str)
@click.argument("model", type=str, required=False)
@click.option("--output-format", "-of", type=str, default=None, help="Output format, when output is media. Valid options depend on the media type. Defaults are png for images, mp4 for videos, and wav for audio.", show_default=True)
@click.option("--model-offload/--no-model-offload", default=False, is_flag=True, show_default=True, help="Offload models to CPU after use in supported pipelines.")
@click.option("--sequential-offload/--no-sequential-offload", default=False, is_flag=True, show_default=True, help="Offload layers to CPU after use in supported pipelines.")
@click.option("--encode-tiling/--no-encode-tiling", default=False, is_flag=True, show_default=True, help="Enable tiled encoding in supported pipelines.")
@click.option("--encode-slicing/--no-encode-slicing", default=False, is_flag=True, show_default=True, help="Enable sliced encoding in supported pipelines.")
@click.option("--context-length", "-cl", default=None, type=int, help="Context length for supported pipelines.", show_default=True)
@click.option("--open-output/--no-open-output", default=True, is_flag=True, show_default=True, help="Open an output file after completion. Only applies to tasks that produce output files.")
@click.option("--json", "-j", is_flag=True, help="Output result as JSON.")
@context_options()
def invoke(
    task: str,
    model: Optional[str]=None,
    output_format: Optional[str]=None,
    model_offload: bool=False,
    sequential_offload: bool=False,
    encode_tiling: bool=False,
    encode_slicing: bool=False,
    context_length: Optional[int]=None,
    open_output: bool=True,
    json: bool=False,
    save_dir: str=DEFAULT_SAVE_DIR,
    model_dir: str=DEFAULT_MODEL_DIR,
    log_level: str=DEFAULT_LOG_LEVEL,
    add_import: List[str]=[],
    quiet: bool=False
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
                "save_dir", "model_dir", "model",
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

    with get_command_context(log_level, add_import, quiet):
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
        click.echo(yellow("Loading task."))
        task_instance = task_class()
        task_instance.save_dir = save_dir
        task_instance.model_dir = model_dir
        task_instance.enable_model_offload = model_offload
        task_instance.enable_sequential_offload = sequential_offload
        task_instance.enable_encode_tiling = encode_tiling
        task_instance.enable_encode_slicing = encode_slicing
        task_instance.context_length = context_length
        task_instance.use_tqdm = True

        with time_counter() as timer:
            task_instance.load()

        click.echo(f"Task loaded in {cyan(human_duration(float(timer)))}.")

        if "output_format" in task_parameters:
            invoke_args["output_format"] = output_format
        if "output_upload" in task_parameters:
            invoke_args["output_upload"] = True

        click.echo(yellow("Invoking task."))
        with time_counter() as timer:
            result = task_instance(**invoke_args)

        click.echo(f"Task invoked in {cyan(human_duration(float(timer)))}. Result:")
        if json:
            import json as pyjson
            click.echo(pyjson.dumps(result))
        else:
            click.echo(green(result))

        with time_counter() as timer:
            task_instance.unload()

        click.echo(f"Task unloaded in {cyan(human_duration(float(timer)))}.")

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
