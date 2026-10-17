"""RhizAeon PA-FF v2.1 — Phylo-Attention Force Field Recombination Scanner.

Thin Python sister over the native Rust engine (``rhizaeon._native``).
The ``rhizaeon`` console script forwards ``sys.argv`` verbatim to the clap
parser inside the extension — the CLI surface and output are identical to
the former standalone Rust binary.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("rhizaeon")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+local"
