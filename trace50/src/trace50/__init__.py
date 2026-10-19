"""TRACE-5.0 — Dynamic Molecular Surveillance of Viral Transmission Networks.

Thin Python sister over the native Rust engine (``trace50._native``).
The ``trace50`` console script forwards ``sys.argv`` verbatim to the clap
parser inside the extension — the CLI surface and output are identical to
the former standalone Rust binary.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("trace50")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+local"
