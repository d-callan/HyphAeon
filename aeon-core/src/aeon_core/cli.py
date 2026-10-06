"""
aeon_core/cli.py
----------------
Shared CLI plumbing for Aeon-family command-line front ends.
"""

import functools
import sys

from .weights import DEFAULT_VARIANT, default_variant, default_weights

# User-facing failures (bad weights path, incompatible weights file/extension,
# missing runtime) rendered as a clean message + exit code instead of a
# traceback. Anything outside these is a code bug and keeps its traceback.
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


def add_weights_args(parser, weights_envs=("HYPHAEON_WEIGHTS",),
                     variant_envs=("HYPHAEON_VARIANT",)):
    """Attach the standard -w/--weights and --model-variant flags to a parser.

    Every Aeon-family subcommand that needs model weights MUST use this —
    hand-declaring the pair per subparser is how flag names/defaults/env-var
    precedence drifted previously. `*_envs` are ordered precedence tuples:
    the first env var names the package's own override, the rest are
    fallbacks (e.g. chronaeon passes ("CHRONAEON_WEIGHTS",
    "HYPHAEON_WEIGHTS") so pre-refactor users' vars keep working).
    """
    parser.add_argument(
        "-w", "--weights",
        default=default_weights(weights_envs[0], weights_envs[1:]),
        help=("Path to local model weights file (overrides Hugging Face "
              f"download). Can also be set via {'/'.join(weights_envs)}."),
    )
    parser.add_argument(
        "--model-variant", dest="variant",
        default=default_variant(variant_envs[0], variant_envs[1:]),
        help=(f"Model variant to download from Hugging Face "
              f"(default: {DEFAULT_VARIANT}). Can also be set via "
              f"{'/'.join(variant_envs)}."),
    )


def weights_kwargs(args) -> dict:
    """Forward the standard --weights/--model-variant flags to the shared
    weights contract.

    resolve_weights_path, load_model, and every run_* pipeline take
    (weights, variant) — splat this (``**weights_kwargs(args)``) instead of
    hand-listing kwargs so a parsed flag can never be silently dropped at a
    call site. Raises AttributeError if the parser didn't get
    add_weights_args, so mis-wiring fails loudly, not silently.
    """
    return {"weights": args.weights, "variant": args.variant}
