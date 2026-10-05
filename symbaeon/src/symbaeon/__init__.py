"""
symbaeon
--------
Host-pathogen coevolution analysis: cross-alignment coselection on a shared
individual axis, epistasis, and interaction landscapes.

Sits on top of hyphaeon (per-alignment scoring) and aeon-core (model + IO).
"""

__version__ = "0.1.0"

from .pairing import (
    PairedSide,
    load_manifest,
    load_paired_alignment,
    build_individual_axis,
)
from .coselect import run_correlate
from .model import run_coselect

__all__ = [
    "PairedSide",
    "load_manifest",
    "load_paired_alignment",
    "build_individual_axis",
    "run_correlate",
    "run_coselect",
]
