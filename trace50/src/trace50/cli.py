"""Console-script entry point: forward ``sys.argv`` to the native engine."""

import sys


def main() -> None:
    """``trace50`` command — transmission network deconvolution pipeline."""
    from trace50._native import trace50_cli

    # argv[0] is normalized so clap's usage/help text shows the tool name
    # rather than a script path (python -m trace50 would print __main__.py).
    sys.exit(trace50_cli(["trace50", *sys.argv[1:]]))
