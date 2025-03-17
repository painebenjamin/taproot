import os
import re
import sys

from setuptools import find_packages, setup

deps = [
    "aioconsole",
    "async-lru",
    "click",
    "dbgpu[fuzz]",
    "docstring_parser",
    "msgpack",
    "omegaconf",
    "packaging",
    "pillow",
    "psutil",
    "pycryptodome",
    "requests",
]

websocket_deps = [
    "websockets",
]

http_deps = [
    "uvicorn",
    "aiohttp",
    "aiodns",
    "starlette",
]

av_deps = [
    "pyav",
    "moviepy<2.0",
]

tool_deps = [
    "pytz",
    "duckduckgo_search",
    "beautifulsoup4",
]

cli_deps = [
    "tabulate",
    "termcolor",
    "tqdm"
]

uv_deps = [
    "uvloop>=0.18",
]

jp_deps = [
    "kanjize",
    "sudachipy",
    "sudachidict_full",
]

setup(
    name="taproot",
    version="0.3.8",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
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
        "ws": websocket_deps,
        "cli": cli_deps,
        "http": http_deps,
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
