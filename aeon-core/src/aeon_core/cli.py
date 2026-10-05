"""
aeon_core/cli.py
----------------
Shared CLI plumbing for Aeon-family command-line front ends.
"""

import functools
import sys

# User-facing failures (bad weights path, incompatible weights file, missing
# runtime) rendered as a clean message + exit code instead of a traceback.
# Anything outside these is a code bug and keeps its traceback.
CLI_ERROR_TYPES = (FileNotFoundError, RuntimeError)


def handle_cli_errors(func):
    """Decorate a CLI main() to render common user errors as a clean [!] exit.

    Keeps per-subcommand try/except blocks for user errors unnecessary:
    a FileNotFoundError raised anywhere under main() (e.g. a bad --weights
    path reaching aeon_core.weights.resolve_weights_path) exits as a
    one-line message rather than a Python traceback.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CLI_ERROR_TYPES as e:
            print(f"\n[!] {e}", file=sys.stderr)
            sys.exit(1)
    return wrapper
