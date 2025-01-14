from ..payload import RequiredLibrary

__all__ = [
    "LIBRARY_ESPEAK",
]

LIBRARY_ESPEAK: RequiredLibrary = {
    "name": "espeak-ng",
    "aliases": ["espeak"],
    "yum": "espeak-ng",
    "apt": "espeak-ng",
    "dnf": "espeak-ng",
    "win": "https://github.com/espeak-ng/espeak-ng/releases",
}
