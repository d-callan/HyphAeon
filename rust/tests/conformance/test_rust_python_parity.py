"""Cross-language conformance tests: Rust trace50 vs Python aeon-core/chronaeon.

A small Rust driver (`cargo build --release --bin conformance`) dumps the
pipeline's intermediate quantities — parsed dates, the TN93 distance matrix,
and the autoclock community tree — as JSON. This suite recomputes each with
the Python implementations and pins the OBSERVED per-engine outputs.

Design note: the two codebases implement related but distinct algorithms and
header contracts, so cross-language equality is NOT generally asserted.
Where the engines genuinely agree (near-identity TN93 distances, ISO dates
reachable by Rust's first-token scan) tight tolerances are used; where they
diverge (saturation ceilings, decimal-year regex coverage, autoclock split
criteria) each engine's current output is pinned as a golden so any drift —
intentional or accidental — fails loudly during the Phase-2 refactor.

Known intentional divergences (audited, NOT parity bugs — do not "fix"):
  * Saturation ceilings and decimal-year regex coverage differ between the
    Rust and Python TN93/date kernels.

AUTOCLOCK UNIFICATION (Phase A, canonical = Rust per Sergei 2026-10-13):
  chronaeon's fit_clock / recursive_spectral_autoclock delegate to
  aeon_core._native -> aeon-autoclock whenever the extension is
  importable, so both engines emit the same partitions and fit
  statistics. The historical divergences (Bartlett n_eff df penalty vs
  n - 2, relative- vs fixed-horizon tier taxonomies, n < 3 leaf field
  zeroing, and the Welch-z cohesive stop vs the modularity/conductance
  gate) survive only in the pure-Python fallback used when the extension
  is absent — those paths are legacy, not canonical.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

RUST_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = RUST_ROOT / "tests" / "fixtures"
REPO_ROOT = RUST_ROOT.parent
CONFORMANCE_BIN = RUST_ROOT / "target" / "release" / "conformance"

sys.path.insert(0, str(REPO_ROOT / "aeon-core" / "src"))
sys.path.insert(0, str(REPO_ROOT / "chronaeon" / "src"))

from aeon_core import _native as aeon_py  # noqa: E402

DEMO_OUTBREAK = FIXTURES / "demo_outbreak.fasta"
HCV_CHRONIC = FIXTURES / "demo_hcv_chronic.fasta"
PAIR_TEST = FIXTURES / "pair_test.fasta"


def _ensure_conformance_bin() -> Path:
    if not CONFORMANCE_BIN.exists():
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "conformance"],
            cwd=RUST_ROOT, check=True,
        )
    return CONFORMANCE_BIN


def _run_driver(fasta: Path, tmp_path: Path) -> dict:
    out = tmp_path / (fasta.stem + ".json")
    subprocess.run(
        [str(_ensure_conformance_bin()), str(fasta), str(out)], check=True
    )
    return json.loads(out.read_text())


def _read_fasta(path: Path):
    names, seqs, cur = [], [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if names and cur is not None:
                seqs.append("".join(cur))
            names.append(line[1:].split()[0])
            cur = []
        else:
            cur.append(line)
    seqs.append("".join(cur))
    return names, seqs


# ---------------------------------------------------------------------------
# TN93 distance matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fasta", [DEMO_OUTBREAK, HCV_CHRONIC])
def test_tn93_matrix_parity(fasta, tmp_path):
    from aeon_core.dataset import compute_tn93_distance_matrix

    rust = _run_driver(fasta, tmp_path)
    names, seqs = _read_fasta(fasta)
    seq_dict = dict(zip(names, seqs))

    py_mat = compute_tn93_distance_matrix(seq_dict, names)
    n = rust["n"]
    rust_mat = np.array(rust["distance_matrix_flat"]).reshape(n, n)

    mask = py_mat >= 0
    assert mask.all(), "python TN93 left unresolved pairs on fixture"

    # Close pairs (d <= 2%): engines agree to within float32 quantization
    # of the Python wrapper's output matrix (observed max diff 6.7e-8).
    close = mask & (py_mat <= 0.02)
    np.testing.assert_allclose(
        rust_mat[close], py_mat[close], rtol=0, atol=1e-6,
        err_msg=f"TN93 close-pair parity failed on {fasta.name}",
    )

    # Moderately divergent pairs: observed gap is 1.01e-4 rel (outbreak,
    # d~0.07) and 3.56e-4 rel (hcv, d~0.35) between the Rust closed-form
    # estimator and the Python tn93 package. Pin with headroom; growth is a
    # regression.
    divergent = mask & ~close & (py_mat < 1.0)
    if divergent.any():
        np.testing.assert_allclose(
            rust_mat[divergent], py_mat[divergent], rtol=5e-4, atol=1e-6,
            err_msg=f"TN93 divergent-pair drift beyond pinned gap on {fasta.name}",
        )


def test_tn93_saturation_unified(tmp_path):
    """W9: both engines now return the canonical Rust finite estimate.

    Pre-W9 the Python `tn93` package capped unresolvable distances at its
    1.0 ceiling while the Rust closed-form estimator returned d=0.7535 for
    this MASE pair. With Python delegating to aeon_py, the divergence is
    gone — both sides must agree on the finite estimator.
    """
    from aeon_core.dataset import compute_tn93_distance_matrix

    rust = _run_driver(PAIR_TEST, tmp_path)
    names, seqs = _read_fasta(PAIR_TEST)
    py_mat = compute_tn93_distance_matrix(dict(zip(names, seqs)), names)
    n = rust["n"]
    rust_mat = np.array(rust["distance_matrix_flat"]).reshape(n, n)

    assert abs(rust_mat[0, 1] - 0.753507) < 1e-5, (
        f"rust saturated-pair estimate drifted: {rust_mat[0, 1]}"
    )
    assert abs(float(py_mat[0, 1]) - rust_mat[0, 1]) < 1e-6, (
        f"python no longer unified with rust kernel: {py_mat[0, 1]}"
    )


def test_tn93_unresolvable_pairs_imputed():
    """Zero-valid-site pairs impute to max distance, never 0 (identical).

    The aeon_py binding returns NaN for pairs with no valid comparison
    sites (all-N/gap sequences); dataset.py must impute them to
    max(1.0, max observed) on both the NxN and NxM paths. Previously the
    Rust path silently fused incomparable taxa as identical (d=0.0).
    """
    from aeon_core.dataset import (
        compute_tn93_cross_distance_matrix,
        compute_tn93_distance_matrix,
    )

    seq_dict = {"a": "ACGTACGT", "b": "ACGTTCGT", "bad": "NNNNNNNN"}
    taxa = ["a", "b", "bad"]

    mat = compute_tn93_distance_matrix(seq_dict, taxa)
    assert np.isfinite(mat).all(), "imputation left NaN in distance matrix"
    assert mat[0, 1] > 0.0, "a-b should be a real positive distance"
    assert mat[0, 2] >= 1.0 and mat[2, 0] >= 1.0
    assert mat[1, 2] >= 1.0 and mat[2, 1] >= 1.0

    cross = compute_tn93_cross_distance_matrix(seq_dict, ["a", "bad"], ["a"])
    assert cross[0, 0] == 0.0  # self-pair
    assert cross[1, 0] >= 1.0  # unresolvable -> imputed max


# ---------------------------------------------------------------------------
# Date parsing — pinned per-header contract table
# ---------------------------------------------------------------------------

# (rust_value, python_value) per header; None = engine returns no date.
# Rust scans delimited tokens left-to-right and takes the FIRST in-range
# numeric/ISO token. Python `parse_header_timestamp` prefers ISO/decimal
# regexes over bare year tokens and requires 2-4 fractional digits for
# decimal years.
EXPECTED_DATES_OUTBREAK = {
    # Python rejects single-decimal-place years (\d{4}\.\d{2,4} pattern);
    # Rust's float-token scan accepts them. All Python values are NaN here,
    # and Rust takes the accession year (first token), not the |suffix.
    "USA_2015_01|2015.0": (2015.0, None),
    "USA_2015_02|2015.5": (2015.0, None),
    "USA_2016_03|2016.2": (2016.0, None),
    "USA_2016_04|2016.8": (2016.0, None),
    "USA_2017_05|2017.3": (2017.0, None),
    "RE-ACCESSION_2013|2013.2": (2013.0, None),
    "RE-ACCESSION_2017|2017.2": (2017.0, None),
    "PASSENGER_BRIDGE|2015.8": (2015.8, None),  # no earlier year token
    "DYAD_2016_A|2016.1": (2016.0, None),
    "DYAD_2016_B|2016.7": (2016.0, None),
}

EXPECTED_DATES_HCV = {
    # Pure ISO suffix with no earlier year token: both engines parse it.
    # Rust uses (month-1)/12 + (day-1)/365.25; Python uses day-of-year —
    # same date, different fractional-year convention (~4e-3 apart).
    "HCV_PWID_01|2017-03-15": (2017.2049965776866, 2017.2),
    "HCV_PWID_02|2017-10-20": (2017.8020191649555, 2017.8),
    "HCV_PWID_03|2018-05-12": (2018.3634496919917, 2018.358904109589),
    "HCV_PWID_04|2019-02-18": (2019.1298767967146, 2019.131506849315),
    # First-token-wins divergence: Rust takes the embedded '2015'/'2019'
    # before reaching the ISO date; Python prefers the ISO date.
    "HCV_REPEAT_2015|2015-01-10": (2015.0, 2015.0246575342467),
    "HCV_REPEAT_2019|2019-01-10": (2019.0, 2019.0246575342467),
    "HCV_DYAD_A|2018-02-14": (2018.118925393566, 2018.1205479452055),
    "HCV_DYAD_B|2018-08-22": (2018.640828199863, 2018.6383561643836),
    "HCV_COMMUNITY_CTRL|2017-06-30": (2017.4960643394936, 2017.4931506849316),
}


@pytest.mark.parametrize(
    "fasta,expected",
    [(DEMO_OUTBREAK, EXPECTED_DATES_OUTBREAK), (HCV_CHRONIC, EXPECTED_DATES_HCV)],
)
def test_date_parsing_pinned_contract(fasta, expected, tmp_path):
    from chronaeon.dating_io import parse_header_timestamp

    rust = _run_driver(fasta, tmp_path)
    assert set(rust["names"]) == set(expected)

    for name, rust_d in zip(rust["names"], rust["dates"]):
        exp_rust, exp_py = expected[name]
        assert rust_d == pytest.approx(exp_rust, abs=1e-9), (
            f"rust date drifted on {name}: {rust_d}"
        )
        py_d = parse_header_timestamp(name)
        if exp_py is None:
            assert math.isnan(py_d), (
                f"python now parses {name} -> {py_d}; update contract table"
            )
        else:
            assert py_d == pytest.approx(exp_py, abs=1e-9), (
                f"python date drifted on {name}: {py_d}"
            )


# ---------------------------------------------------------------------------
# Autoclock spectral deconvolution — per-engine partition goldens
# ---------------------------------------------------------------------------

# Pinned partitions (P0.4): terminal leaf communities on demo_outbreak.fasta.
# Phase A unified the engines: with aeon_core._native present, Python
# delegates to aeon-autoclock and MUST reproduce the Rust partition
# bit-for-bit. The legacy pure-Python golden is kept only to pin the
# fallback semantics when the extension is absent.
# W11 note: the Rust partition was re-pinned after porting five contract
# gaps (min_size on children, isolated-node tolerance, canonical percentile
# for sigma, positive-mu tMRCA, lambda_2 homogeneity gate). The old {8,9}
# dyad leaf violated the min_size=3 contract and is correctly absorbed.
EXPECTED_PARTITION_RUST = frozenset(
    [frozenset({0, 1, 5, 6}), frozenset({2, 3, 4}), frozenset({7, 8, 9})]
)
EXPECTED_PARTITION_PYTHON_LEGACY = frozenset(
    [frozenset({0, 2, 4, 6, 7}), frozenset({1, 3, 5, 8, 9})]
)


def test_stats_kernels_parity():
    """W12: aeon-manifold stats kernels must match the scipy formulas.

    aeon_core.stats prefers aeon_py and falls back to the scipy
    implementation; this pins their agreement on edge + interior cases.
    """
    from scipy.stats import chi2

    lrts = np.array([0.0, -0.5, 1e-4, 1.0, 3.8415, 25.0])

    # MEME mixture null: 1/3 delta(0) + 2/3 (0.45 chi2_1 + 0.55 chi2_2)
    ref_meme = np.full(len(lrts), 2.0 / 3.0)
    pos = lrts > 0.0
    ref_meme[pos] = (2.0 / 3.0) * (
        0.45 * chi2.sf(lrts[pos], df=1) + 0.55 * chi2.sf(lrts[pos], df=2)
    )
    np.testing.assert_allclose(
        aeon_py.pvals_from_lrt_meme(lrts.tolist()), ref_meme, rtol=1e-12
    )

    # Self & Liang null: 0.5 delta(0) + 0.5 chi2_1
    ref_sl = np.ones(len(lrts))
    ref_sl[pos] = 0.5 * chi2.sf(lrts[pos], df=1)
    np.testing.assert_allclose(
        aeon_py.pvals_from_lrt_self_liang(lrts.tolist()), ref_sl, rtol=1e-12
    )

    # BH on a non-trivial ranking (duplicate p-values exercise ties)
    pv = np.array([0.01, 0.04, 0.03, 0.002, 0.04, 1.0])
    order = np.argsort(pv)
    sorted_p = pv[order]
    q_sorted = np.minimum.accumulate(
        (sorted_p * len(pv) / (np.arange(len(pv)) + 1))[::-1]
    )[::-1]
    ref_bh = np.empty_like(pv)
    ref_bh[order] = np.clip(q_sorted, 0.0, 1.0)
    np.testing.assert_allclose(
        aeon_py.benjamini_hochberg(pv.tolist()), ref_bh, atol=1e-15
    )

    # Cauchy combination vs direct formula
    p_clip = np.clip(pv, 1e-15, 1.0 - 1e-15)
    ref_cct = float(
        np.clip(0.5 - np.arctan(np.mean(np.tan((0.5 - p_clip) * np.pi))) / np.pi,
                1e-15, 1.0)
    )
    assert abs(aeon_py.cauchy_combination_p(pv.tolist()) - ref_cct) < 1e-15
    assert aeon_py.cauchy_combination_p([]) == 1.0


def test_stats_kernels_nan_contract():
    """NaN inputs follow the scipy/numpy reference semantics exactly.

    `pos = lrts > 0.0` masks NaN into the point-mass branch; `np.argsort`
    places NaN last and `np.minimum.accumulate` poisons the BH cumulative
    min, so one NaN p-value yields all-NaN q-values.
    """
    from scipy.stats import chi2

    lrts = [float("nan"), -1.0, 4.0]
    pos = np.array(lrts) > 0.0

    ref_meme = np.full(3, 2.0 / 3.0)
    ref_meme[pos] = (2.0 / 3.0) * (
        0.45 * chi2.sf(4.0, df=1) + 0.55 * chi2.sf(4.0, df=2)
    )
    np.testing.assert_allclose(
        aeon_py.pvals_from_lrt_meme(lrts), ref_meme, rtol=1e-12
    )

    ref_sl = np.ones(3)
    ref_sl[pos] = 0.5 * chi2.sf(4.0, df=1)
    np.testing.assert_allclose(
        aeon_py.pvals_from_lrt_self_liang(lrts), ref_sl, rtol=1e-12
    )

    # BH: replicate the Python reference then assert all-NaN output parity
    pv = np.array([0.01, np.nan, 0.04])
    order = np.argsort(pv)
    sorted_p = pv[order]
    q_sorted = np.minimum.accumulate(
        (sorted_p * len(pv) / (np.arange(len(pv)) + 1))[::-1]
    )[::-1]
    ref_bh = np.empty_like(pv)
    ref_bh[order] = np.clip(q_sorted, 0.0, 1.0)
    rust_bh = np.array(aeon_py.benjamini_hochberg(pv.tolist()))
    assert np.isnan(ref_bh).all(), "numpy reference itself changed"
    assert np.isnan(rust_bh).all(), (
        f"rust BH NaN handling diverged from numpy: {rust_bh}"
    )


def test_autoclock_partition_unified(tmp_path):
    """Post-Phase A: the delegated Python path IS the canonical Rust engine.

    Feeds the Rust-parsed dates and Rust-computed TN93 matrix to both the
    conformance driver and chronaeon's recursive_spectral_autoclock. When
    aeon_core._native is importable the Python call delegates to
    aeon-autoclock, so the partitions must be IDENTICAL — not merely
    similar. When the extension is absent the legacy pure-Python fallback
    keeps its own pinned golden.
    """
    import chronaeon.autoclock as ch_ac

    rust = _run_driver(DEMO_OUTBREAK, tmp_path)
    n = rust["n"]
    d_mat = np.array(rust["distance_matrix_flat"]).reshape(n, n)
    dates = np.array([d if d is not None else np.nan for d in rust["dates"]])

    rust_part = frozenset(frozenset(c["members"]) for c in rust["communities"])
    leaves = ch_ac.recursive_spectral_autoclock(
        np.arange(n),
        dates,
        d_mat,
        min_size=3,
        max_depth=10,
    )
    py_part = frozenset(frozenset(leaf[0].tolist()) for leaf in leaves)

    # Invariants both engines must satisfy regardless of split criteria:
    for part, engine in ((rust_part, "rust"), (py_part, "python")):
        flat = sorted(m for c in part for m in c)
        assert flat == list(range(n)), f"{engine} partition not a cover of taxa"

    assert rust_part == frozenset(EXPECTED_PARTITION_RUST), (
        f"rust autoclock partition drifted: "
        f"{sorted(map(sorted, rust_part))}"
    )
    if ch_ac._native_ext is not None:
        assert py_part == rust_part, (
            "delegated Python autoclock diverged from canonical Rust: "
            f"{sorted(map(sorted, py_part))}"
        )
    else:
        assert py_part == frozenset(EXPECTED_PARTITION_PYTHON_LEGACY), (
            f"python fallback autoclock partition drifted: "
            f"{sorted(map(sorted, py_part))}"
        )
