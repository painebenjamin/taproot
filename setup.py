import os
import re
import sys

from typing import List
from setuptools import find_packages, setup

def read_requirements_file(file_name: str) -> List[str]:
    """
    Reads a requirements file and returns a list of dependencies.
    """
    with open(f"./{file_name}.txt", "r", encoding="utf-8") as file:
        requirements = file.readlines()

    # Remove comments and empty lines
    requirements: List[str] = []
    for line in requirements:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements

deps = read_requirements_file("requirements")

av_deps = read_requirements_file("extras-av")
cli_deps = read_requirements_file("extras-cli")
http_deps = read_requirements_file("extras-http")
jp_deps = read_requirements_file("extras-jp")
mypy_deps = read_requirements_file("extras-mypy")
tool_deps = read_requirements_file("extras-tools")
uv_deps = read_requirements_file("extras-uv")
ws_deps = read_requirements_file("extras-ws")

setup(
    name="taproot",
    version="0.3.9",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="Taproot is a seamlessly scalable AI/ML inference engine designed for deployment across hardware clusters with disparate capabilities.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Benjamin Paine",
    author_email="painebenjamin@gmail.com",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"taproot": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=deps,
    extras_require={
        "av": av_deps,
        "uv": uv_deps,
        "jp": jp_deps,
        "ws": ws_deps,
        "cli": cli_deps,
        "http": http_deps,
        "mypy": mypy_deps,
        "tools": tool_deps,
    },
    entry_points={
        "console_scripts": [
            "taproot = taproot.__main__:main"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}" for i in range(8, 13)],
)
