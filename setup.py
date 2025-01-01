import os
import re
import sys

from setuptools import find_packages, setup

deps = [
    "async-lru",
    "click",
    "dbgpu[fuzz]",
    "docstring_parser",
    "httptools",
    "msgpack",
    "nest_asyncio",
    "omegaconf",
    "packaging",
    "pillow",
    "psutil",
    "pycryptodome",
    "requests",
    "websockets",
]

video_deps = [
    "pyav",
    "moviepy<2.0",
]

tool_deps = [
    "pytz",
    "duckduckgo_search",
    "beautifulsoup4",
]

console = [
    "tabulate",
    "termcolor",
    "tqdm"
]

setup(
    name="taproot",
    version="0.1.0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
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
        "video": video_deps,
        "tools": tool_deps,
        "console": console,
    },
    entry_points={
        "console_scripts": [
            "taproot = taproot.__main__:main"
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
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
