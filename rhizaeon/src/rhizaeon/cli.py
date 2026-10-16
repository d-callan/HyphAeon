"""Console-script entry points: forward ``sys.argv`` to the native engine."""

import sys


def main() -> None:
    """``rhizaeon`` command — recombination scan over a FASTA alignment."""
    from rhizaeon._native import rhizaeon_cli

    # argv[0] is normalized so clap's usage/help text shows the tool name
    # rather than a script path (python -m rhizaeon would print __main__.py).
    sys.exit(rhizaeon_cli(["rhizaeon", *sys.argv[1:]]))
