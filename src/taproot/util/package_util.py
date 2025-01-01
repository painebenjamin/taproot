from __future__ import annotations

import os
import sys
import struct
import platform

from packaging.version import Version
from packaging.specifiers import SpecifierSet

from typing import Union, Optional, Dict, List, Tuple

__all__ = [
    "combine_package_specifications",
    "get_combined_specifications",
    "get_installed_package_version",
    "get_llama_cpp_package_versions",
    "get_llama_cpp_package_url",
    "get_pending_packages",
    "get_pip_package_name",
    "get_torch_package_version",
    "get_torch_package_versions",
    "get_torch_packages",
    "install_package",
    "install_packages",
    "installed_package_matches_spec",
    "is_torch_package",
    "version_matches_spec",
]

PREBUILT_CUDA_VERSION = os.getenv("PREBUILT_CUDA_VERSION", "cu121")
NVIDIA_REPO_URL = "https://pypi.ngc.nvidia.com"
TORCH_REPO_URL = f"https://download.pytorch.org/whl/{PREBUILT_CUDA_VERSION}"
LLAMA_CPP_REPO_URL = f"https://abetlen.github.io/llama-cpp-python/whl/{PREBUILT_CUDA_VERSION}"

TORCH_PACKAGES: Optional[List[str]] = None
def get_torch_packages() -> List[str]:
    """
    Get the list of PyTorch packages available for download.

    Only retrieves names, versions must be requested separately.

    :return: List of PyTorch package names.
    """
    global TORCH_PACKAGES
    if TORCH_PACKAGES is None:
        from requests import get
        from xml.etree import ElementTree as ET
        tree = ET.fromstring(get(TORCH_REPO_URL).text)
        TORCH_PACKAGES = [
            elem.text
            for elem in tree[0]
            if elem.text
        ]
    return TORCH_PACKAGES

def is_torch_package(package_name: str) -> bool:
    """
    Check if a package is a PyTorch package.
    """
    return package_name in get_torch_packages()

TORCH_PACKAGE_VERSIONS: Dict[str, Tuple[List[Version], List[Version]]] = {}
def get_torch_package_versions(package_name: str) -> Tuple[List[Version], List[Version]]:
    """
    Get the list of versions for a PyTorch package.

    :param package_name: The name of the PyTorch package.
    :return: A tuple of lists of versions: (base versions, CUDA versions)
    :see: PREBUILT_CUDA_VERSION
    """
    global TORCH_PACKAGE_VERSIONS
    if package_name not in TORCH_PACKAGE_VERSIONS:
        from requests import get
        from xml.etree import ElementTree as ET
        tree = ET.fromstring(get(f"{TORCH_REPO_URL}/{package_name}/").text)
        raw_package_filenames = [elem.text for elem in tree[0]][1:]
        package_versions = set(
            filename[len(package_name)+1:].split("-")[0]
            for filename in raw_package_filenames
            if filename
        )
        base_package_versions = set(
            version
            for version in package_versions
            if "+" not in version
        )
        cuda_package_versions = set(
            version
            for version in package_versions
            if version.endswith(f"+{PREBUILT_CUDA_VERSION}")
        )
        base_package_version_objects = [Version(version) for version in base_package_versions]
        base_package_version_objects.sort()
        cuda_package_version_objects = [Version(version.split("+")[0]) for version in cuda_package_versions]
        cuda_package_version_objects.sort()
        TORCH_PACKAGE_VERSIONS[package_name] = (
            base_package_version_objects,
            cuda_package_version_objects
        )
    return TORCH_PACKAGE_VERSIONS[package_name]

def get_torch_package_version(
    package_name: str,
    version_spec: Optional[str]=None,
    cuda_only: bool=True
) -> Optional[str]:
    """
    Gets the best version for a PyTorch package.

    :param package_name: The name of the PyTorch package.
    :param version_spec: A version specifier string.
    :return: The best version for the package.
    """
    base_versions, cuda_versions = get_torch_package_versions(package_name)
    matching_base_versions = [
        version for version in base_versions
        if version_spec is None or
        version_matches_spec(version, version_spec)
    ]
    matching_cuda_versions = [
        version for version in cuda_versions
        if version_spec is None or
        version_matches_spec(version, version_spec)
    ]
    if matching_cuda_versions:
        # Always prefer CUDA versions
        return f"{matching_cuda_versions[-1]}+{PREBUILT_CUDA_VERSION}"
    if cuda_only:
        return None
    if matching_base_versions:
        return str(matching_base_versions[-1])
    return None

def get_filtered_wheels(wheel_files: List[str]) -> List[str]:
    """
    Gets a list of wheel files that are compatible with the current system.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    is_version_match = lambda wheel_file: python_version in wheel_file

    if system == "windows":
        if "a,d64" in machine or struct.calcsize("P") * 8 == 64:
            is_machine_match = lambda wheel_file: "win" in wheel_file and "amd64" in wheel_file
        else:
            is_machine_match = lambda wheel_file: "win32" in wheel_file
    elif system == "darwin":
        if machine == "x86_64" or machine == "amd64":
            is_machine_match = lambda wheel_file: "macosx" in wheel_file and "x86_64" in wheel_file
        elif machine == "arm64":
            is_machine_match = lambda wheel_file: "macosx" in wheel_file and "arm64" in wheel_file
        else:
            is_machine_match = lambda wheel_file: "macosx" in wheel_file
    elif system == "linux":
        if "x86_64" in machine or "amd64" in machine:
            is_machine_match = lambda wheel_file: "linux" in wheel_file and ("x86_64" in wheel_file or "amd64" in wheel_file)
        elif "aarch64" in machine or "arm64" in machine:
            is_machine_match = lambda wheel_file: "linux" in wheel_file and "aarch64" in wheel_file
        else:
            is_machine_match = lambda wheel_file: "linux" in wheel_file
    else:
        raise ValueError(f"Unsupported system: {system}")

    return [
        wheel_file for wheel_file in wheel_files
        if wheel_file.endswith(".whl")
        and is_version_match(os.path.basename(wheel_file)) # type: ignore[no-untyped-call]
        and is_machine_match(os.path.basename(wheel_file)) # type: ignore[no-untyped-call]
    ]

LLAMA_CPP_PACKAGE_VERSIONS: Optional[Dict[Version, str]] = None
def get_llama_cpp_package_versions() -> Dict[Version, str]:
    """
    Get the list of versions for the llama-cpp-python package.

    :return: A list of versions.
    """
    global LLAMA_CPP_PACKAGE_VERSIONS
    if LLAMA_CPP_PACKAGE_VERSIONS is None:
        from requests import get
        from xml.etree import ElementTree as ET
        content = get(f"{LLAMA_CPP_REPO_URL}/llama-cpp-python").text.replace("<br>", "")
        tree = ET.fromstring(content)
        all_package_urls = [
            elem.text for elem in tree[0]
            if isinstance(elem.text, str)
            and elem.text.endswith(".whl")
        ]
        LLAMA_CPP_PACKAGE_VERSIONS = {}
        for package_url in get_filtered_wheels(all_package_urls):
            package_filename = package_url.split("/")[-1]
            package_basename, package_extension = os.path.splitext(package_filename)
            package_filename, package_version, python_version, abi, platform = package_basename.split("-")
            LLAMA_CPP_PACKAGE_VERSIONS[Version(package_version)] = package_url
    return LLAMA_CPP_PACKAGE_VERSIONS

def get_llama_cpp_package_url(version_spec: Optional[str]=None) -> str:
    """
    Get the URL for a specific version of the llama-cpp-python package.

    :param version: The version of the package.
    :return: The URL of the package.
    """
    llama_cpp_package_versions = get_llama_cpp_package_versions()
    for version in reversed(sorted(llama_cpp_package_versions.keys())):
        if version_spec is None or version_matches_spec(version, version_spec):
            return llama_cpp_package_versions[version]
    raise ValueError(f"No matching version for spec: {version_spec}")

def get_pip_package_name(package_name: str) -> str:
    """
    Gets the package name - some packages have different names on PyPI.

    :param package_name: The package name.
    :return: The PyPI package name.
    """
    package_name = package_name.split("[")[0]
    name_lower = package_name.lower()
    if name_lower == "pil":
        return "Pillow"
    elif name_lower == "sklearn":
        return "scikit-learn"
    elif name_lower == "skimage":
        return "scikit-image"
    elif name_lower == "cv2":
        return "opencv-python"
    elif name_lower == "llama_cpp":
        return "llama-cpp-python"
    elif name_lower == "df":
        return "deepfilternet"
    elif name_lower == "tts":
        return "TTS"
    return package_name

def version_matches_spec(version: Union[str, Version], spec: str) -> bool:
    """
    Check if a version string matches a version spec string.

    >>> version_matches_spec("1.2.3", ">=1.0.0")
    True
    >>> version_matches_spec("1.2.3", ">=1.0.0,<2.0.0")
    True
    >>> version_matches_spec("1.2.3", ">=2.0.0")
    False
    >>> version_matches_spec("1.2.3", ">=2.0.0,<3.0.0")
    False

    :param version: The version to check.
    :param spec: The version spec to check against.
    :return: Whether or not the version matches the spec.
    """
    if not version or not spec:
        return False

    if isinstance(version, str):
        found_version = Version(version)
    else:
        found_version = version

    try:
        version_operators = ["!=", "<=", ">=", "==", "<", ">", "="]
        specs = spec.split(",")
        for spec in specs:
            for operator in version_operators:
                if spec.startswith(operator):
                    spec_version = Version(spec[len(operator):])
                    if operator == "!=":
                        assert spec_version != found_version
                    elif operator == "<=":
                        assert found_version <= spec_version
                    elif operator == ">=":
                        assert found_version >= spec_version
                    elif operator == "<":
                        assert found_version < spec_version
                    elif operator == ">":
                        assert found_version > spec_version
                    elif operator in ["==", "="]:
                        assert found_version == spec_version
                    else:
                        raise ValueError(f"Unknown operator: {operator}")
                    break
    except AssertionError:
        return False

    return True

def get_installed_package_version(package_name: str) -> Version:
    """
    Get the version of an installed package.
    """
    from importlib.metadata import version
    return Version(version(get_pip_package_name(package_name)))

def installed_package_matches_spec(package_name: str, spec: Optional[str]=None) -> bool:
    """
    Check if the version of an installed package matches a version spec string.
    When the spec is None, it will return True if the package is installed.
    When the package is not installed, it will return False.

    >>> installed_package_matches_spec("wheel")
    True
    >>> installed_package_matches_spec("wheel", ">=0.0.1")
    True
    >>> installed_package_matches_spec("wheel", ">=0.0.1,<10")
    True
    >>> installed_package_matches_spec("wheel", ">=10.0.0")
    False
    >>> installed_package_matches_spec("something_that_doesnt_exist", ">=0.0.1")
    False
    """
    try:
        package_version = get_installed_package_version(package_name)
    except Exception:
        return False
    if spec is None:
        return True
    return version_matches_spec(package_version, spec)

def package_is_installed(package_name: str) -> bool:
    """
    Check if a package is installed.

    A shorthand for `installed_package_matches_spec(package_name, None)`.

    Being 'installed' does not necessarily mean 'available.' (importable)
    """
    return installed_package_matches_spec(package_name)

def get_combined_specifications(*versions: Optional[str]) -> Optional[str]:
    """
    Combine multiple version specifications into a single string.
    """
    spec_set = SpecifierSet()
    has_spec = False

    for version in versions:
        if version is not None:
            has_spec = True
            spec_set &= SpecifierSet(version)
    if not has_spec:
        return None
    return str(spec_set)

def combine_package_specifications(*specs: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """
    Combine multiple package specifications into a single dictionary.
    """
    all_specs: Dict[str, List[Optional[str]]] = {}
    for spec in specs:
        for package_name, package_spec in spec.items():
            if package_name not in all_specs:
                all_specs[package_name] = []
            all_specs[package_name].append(package_spec)
    combined_specs: Dict[str, Optional[str]] = {}
    for package_name, versions in all_specs.items():
        try:
            combined_specs[package_name] = get_combined_specifications(*versions)
        except ValueError:
            raise ValueError(f"Invalid version spec(s) for package: {package_name} ({versions})")
    return combined_specs

def get_pending_packages(packages: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """
    Get a list of packages that are not installed or do not match the specified version spec.
    """
    pending_packages = {}
    for package_name, spec in packages.items():
        if not installed_package_matches_spec(package_name, spec):
            pending_packages[package_name] = spec
    return pending_packages

def install_mim_packages(args: List[str]) -> int:
    """
    Install packages using MIM.
    """
    from mim.commands.install import install # type: ignore[import-untyped,import-not-found,unused-ignore]
    return install(args) # type: ignore[no-any-return]

def install_pip_packages(args: List[str]) -> int:
    """
    Install packages using pip.
    """
    from pip._internal.commands import create_command
    return create_command("install").main(args)

def is_mim_package(package_name: str) -> bool:
    """
    Returns whether or not the package is one that is installed by MIM.
    """
    return package_name.split("-")[0].lower() in [
        "mmcv", "mmdet", "mmseg", "mmsegmentation",
        "mmocr", "mmaction", "mmcls", "mmengine",
        "mmagic", "mmyolo", "mmrotate", "mmtracking",
        "mmhuman3d", "mmselfsup", "mmfewshot", "mmaction2",
        "mmflow", "mmdeploy", "mmrazor"
    ]

def install_packages(
    packages: Dict[str, Optional[str]],
    reinstall: bool=False
) -> None:
    """
    Install packages if they are not already installed.
    """
    from .log_util import logger
    if not packages:
        logger.debug("No packages to install.")
        return

    use_mim = False 
    name_specs: Dict[str, str] = {}
    for package_name, spec in packages.items():
        package_name = get_pip_package_name(package_name)
        if package_name =="llama-cpp-python":
            try:
                package_url = get_llama_cpp_package_url(spec)
                # Swap the package name for the URL
                name_specs[package_url] = ""
                continue
            except Exception as e:
                logger.warning(f"Could not find pre-built package for {package_name}{spec or ''}: {e}")
                # Build the package from source
                logger.debug("Setting CMAKE_ARGS environment variable to enable CUDA for llama-cpp-python")
                os.environ["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        if is_mim_package(package_name):
            use_mim = True
        elif is_torch_package(package_name):
            new_spec = get_torch_package_version(package_name, spec, cuda_only=True)
            if new_spec is not None:
                logger.debug(f"Using PyTorch version {new_spec} for package {package_name}{spec or ''}")
                spec = f"=={new_spec}"

        name_specs[package_name] = spec or ""

    if use_mim:
        # Mim will break if torch is not installed first, so we check for it here
        torch_installed = package_is_installed("torch")
        if not torch_installed:
            # We need to install torch first
            torch_spec = name_specs.get("torch", None)
            install_package("torch", torch_spec)

        # Make sure openmim is installed as well
        openmim_installed = package_is_installed("openmim")
        if not openmim_installed:
            openmim_spec = name_specs.pop("openmim", None)
            install_package("openmim", openmim_spec)

    # Install ninja before installing any packages that may benefit from it
    install_ninja = "ninja" in name_specs
    if install_ninja and len(name_specs) > 1:
        ninja_installed = package_is_installed("ninja")
        if not ninja_installed:
            ninja_spec = name_specs.pop("ninja", None)
            install_package("ninja", ninja_spec)

    main_args = [
        f"{k}{v}" for k, v in name_specs.items()
    ] + [
        chunk for REPO_URL in [NVIDIA_REPO_URL, TORCH_REPO_URL, LLAMA_CPP_REPO_URL]
        for chunk in ["--extra-index-url", REPO_URL]
    ]

    if reinstall:
        main_args.append("--ignore-installed")

    logger.debug(f"Installing {'mim' if use_mim else 'pip'} packages with command `{' '.join(main_args)}`")

    if use_mim:
        status_code = install_mim_packages(main_args)
    else:
        status_code = install_pip_packages(main_args)

    assert status_code == 0, "Failed to install packages"

def install_package(package_name: str, spec: Optional[str]=None) -> None:
    """
    Install a package if it is not already installed.
    """
    if installed_package_matches_spec(package_name, spec):
        return
    install_packages({package_name: spec})
