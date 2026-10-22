"""
aeon_core/cli.py
----------------
Shared CLI plumbing for Aeon-family command-line front ends.
"""

import functools
import os
import sys

from .model import DEFAULT_MAX_MEMORY_MB
from .weights import (
    DEFAULT_VARIANT, WeightsError, default_variant, default_weights,
)

# User-facing failures (bad weights path, incompatible weights file/extension,
# missing executable) rendered as a clean message + exit code instead of a
# traceback. Deliberately NOT bare RuntimeError — torch raises that for CUDA
# OOM, shape mismatches in forward, device errors etc., which are bugs or
# environment faults whose traceback must be preserved. Weights-domain user
# errors are WeightsError instead.
CLI_ERROR_TYPES = (FileNotFoundError, WeightsError)


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


def _default_max_memory_mb(env_var: str, fallback_env_vars) -> float:
    """Default --max-memory-mb for a package CLI (env var -> fallbacks -> default)."""
    for var in (env_var,) + tuple(fallback_env_vars):
        value = os.environ.get(var)
        if value:
            try:
                return float(value)
            except ValueError:
                print(f"[!] Ignoring malformed {var}={value!r} (expected a number of MB).",
                      file=sys.stderr)
                continue
    return DEFAULT_MAX_MEMORY_MB


def add_memory_args(parser, envs=("AEON_MAX_MEMORY_MB",)):
    """Attach the standard --max-memory-mb flag to a parser.

    Every Aeon-family subcommand that builds a tree attention cache (or any
    other large quadratic working structure) MUST use this — hand-declaring
    the flag per subparser is how flag names/defaults drifted previously.
    `envs` is an ordered precedence tuple: the first env var names the
    package's own override, the rest are fallbacks.

    The budget caps *cached* quadratic structures (e.g. the all-layers
    num_layers x num_heads x (N+1)^2 Markov bias tensor); when a structure
    would exceed it, the code takes the speed hit and evaluates lazily
    instead of allocating. Transient per-batch attention memory is governed
    separately by the adaptive chunk/batch machinery.
    """
    parser.add_argument(
        "--max-memory-mb", type=float, dest="max_memory_mb",
        default=_default_max_memory_mb(envs[0], envs[1:]),
        help=("Maximum RAM budget (MB) for large cached matrices "
              f"(default: {DEFAULT_MAX_MEMORY_MB:.0f}). Structures exceeding "
              "the budget are evaluated lazily instead of precomputed, trading "
              f"speed for bounded memory. Can also be set via {'/'.join(envs)}."),
    )


def memory_kwargs(args) -> dict:
    """Forward the standard --max-memory-mb flag to cache-building call sites.

    prepare_alignment / precompute_tree_cache take max_memory_mb — splat this
    (``**memory_kwargs(args)``) instead of hand-listing the kwarg so a parsed
    flag can never be silently dropped. Raises AttributeError if the parser
    didn't get add_memory_args, so mis-wiring fails loudly, not silently.
    """
    return {"max_memory_mb": args.max_memory_mb}
