"""TRACE-5.0 — Dynamic Molecular Surveillance of Viral Transmission Networks.

Thin Python sister over the native Rust engine (``trace50._native``).
The ``trace50`` console script forwards ``sys.argv`` verbatim to the clap
parser inside the extension — the CLI surface and output are identical to
the former standalone Rust binary.
"""

__version__ = "5.0.0"
