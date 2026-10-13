"""
aeon_core/stats.py
------------------
Shared statistical functions: LRT → p-value conversion, Benjamini-Hochberg
FDR q-values, and Cauchy Combination Test (CCT).

These are used by the CLI (cmd_meme, cmd_busted), model_eval/_harness.py,
hyphaeon/filter.py, hyphaeon/epistasis.py, and hyphaeon/phenotype.py.
Centralizing them here ensures all call sites use the same formula.

Canonical engine: aeon-manifold::stats (Rust), exposed via aeon_core._native.
The extension is built unconditionally by maturin on any install of this
package, so it is imported unconditionally — a missing extension means a
broken install and should fail loudly rather than silently diverge.
"""

import numpy as np

from aeon_core import _native as aeon_py


def pvals_from_lrt_meme(lrts: np.ndarray) -> np.ndarray:
    """Convert LRTs to p-values using the MEME asymptotic mixture null.

    MEME.bf null: 1/3 * delta(0) + 2/3 * (0.45 * chi2(1) + 0.55 * chi2(2))

    Sites with LRT = 0 get p = 2/3 (the point mass at 0). Sites with LRT > 0
    use the continuous mixture tail.

    This is the formula HyPhy MEME uses and the one HyphAeon's neural model
    is trained to mimic (training targets are MEME LRTs).
    """
    return np.asarray(aeon_py.pvals_from_lrt_meme(np.asarray(lrts, dtype=np.float64).tolist()))


def pvals_from_lrt_self_liang(lrts: np.ndarray) -> np.ndarray:
    """Convert LRTs to p-values using the Self & Liang (1987) mixture null.

    Null: 0.5 * delta(0) + 0.5 * chi2(1)

    Used by BUSTED (whole-gene omnibus test) and by attribution p-values
    (single-parameter counterfactual tests). Not appropriate for per-site
    MEME-style p-values — use pvals_from_lrt_meme for those.
    """
    return np.asarray(aeon_py.pvals_from_lrt_self_liang(np.asarray(lrts, dtype=np.float64).tolist()))


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR q-values from p-values.

    Returns q-values clipped to [0, 1], same length as input.
    """
    return np.asarray(aeon_py.benjamini_hochberg(np.asarray(pvals, dtype=np.float64).tolist()))


def cauchy_combination_p(pvals: np.ndarray) -> float:
    """Compute the Cauchy Combination Test (CCT) omnibus p-value.

    Combines per-site p-values into a single gene-level p-value using the
    Cauchy combination: tan((0.5 - p) * pi) averaged, then back-transformed.
    Robust to dependency among the input p-values.
    """
    return float(aeon_py.cauchy_combination_p(np.asarray(pvals, dtype=np.float64).tolist()))
