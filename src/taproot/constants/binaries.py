from ..payload import RequiredBinary

__all__ = [
    "BINARY_RUST",
]

BINARY_RUST: RequiredBinary = {
    "name": "rust",
    "aliases": ["rustc", "cargo"],
    "yum": "rust",
    "apt": "rustc",
    "dnf": "rust",
    "conda": "rust",
    "win": "https://www.rust-lang.org/tools/install",
}
