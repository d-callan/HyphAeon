"""Direct contract tests for the ``aeon_core._native`` extension.

Everything else in the suite exercises these functions through Python
wrappers; these tests pin the binding-level behavior (argument types,
JSON shapes, sentinel values) that chronaeon/trace50/py callers depend
on. Sources of truth for the documented semantics:

- ``rust/libs/aeon-manifold/src/encoding.rs`` — canonical IUPAC table
- ``rust/libs/aeon-autoclock/src/lib.rs`` — clock fit + spectral recursion
- ``aeon-core/native/src/lib.rs`` — binding docstrings (NaN on zero
  valid TN93 sites, ragged truncation, strict/pad modes)
"""

import json
import math

import pytest

import aeon_core._native as native


# ---- TN93 ---------------------------------------------------------------

def test_tn93_identical_and_symmetric():
    m = native.tn93_distance_matrix([b"ACGTACGT", b"ACGTACGT"])
    assert m == [0.0, 0.0, 0.0, 0.0]

    a, b = b"ACGTACGT", b"ACGAACGA"
    m = native.tn93_distance_matrix([a, b])
    assert m[0] == 0.0 and m[3] == 0.0
    assert m[1] == m[2] and m[1] > 0.0


def test_tn93_zero_valid_sites_returns_nan():
    # 'X' has IUPAC mask 0, so the pair has no valid sites; the kernel
    # returns NaN so callers can impute rather than score identity.
    m = native.tn93_distance_matrix([b"XXXX", b"XXXX"])
    assert m[0] == 0.0 and m[3] == 0.0
    assert math.isnan(m[1]) and math.isnan(m[2])


def test_tn93_ragged_truncates_to_shorter():
    # Historical contract: ragged pairs are compared over min(len).
    m = native.tn93_distance_matrix([b"ACGT", b"ACGTGG"])
    assert m[1] == 0.0


def test_tn93_cross_matches_full_matrix():
    a, b, c = b"ACGTACGT", b"ACGAACGA", b"TTTTTTTT"
    full = native.tn93_distance_matrix([a, b, c])
    cross = native.tn93_cross_distance_matrix([a], [b, c])
    assert cross == [full[1], full[2]]
    # Cross matrix is row-major N x M, not symmetric.
    rev = native.tn93_cross_distance_matrix([b, c], [a])
    assert rev == [full[3], full[6]]


# ---- IUPAC encoding ------------------------------------------------------

def test_iupac_mask_canonical_table():
    m = native.char_to_iupac_mask
    assert m(ord("A")) == 0b0001
    assert m(ord("C")) == 0b0010
    assert m(ord("G")) == 0b0100
    assert m(ord("T")) == 0b1000
    assert m(ord("U")) == 0b1000
    assert m(ord("R")) == 0b0101
    assert m(ord("V")) == 0b0111
    assert m(ord("a")) == m(ord("A"))  # case-insensitive
    # N, gaps, and unknown characters intentionally map to 0
    # (encoding.rs: "N, gap, unknown" -> 0b0000) so TN93 skips the site.
    assert m(ord("N")) == 0
    assert m(ord("-")) == 0
    assert m(ord("X")) == 0


def test_iupac_probabilities_uniform_over_set():
    # R = A or G -> [0.5, 0, 0.5, 0] in [A, C, G, T] order.
    assert native.iupac_base_probabilities(ord("R")) == [0.5, 0.0, 0.5, 0.0]
    # Mask-0 input falls back to the uniform distribution (trace50
    # contract documented in encoding.rs).
    assert native.iupac_base_probabilities(ord("N")) == [0.25, 0.25, 0.25, 0.25]


# ---- FASTA ---------------------------------------------------------------

def test_fasta_tokenize_first_token_names():
    recs = native.fasta_tokenize(">seq_a description here\nACGT\nAC\n>seq_b\nTT\n")
    assert recs == [("seq_a", b"ACGTAC"), ("seq_b", b"TT")]


def test_fasta_normalize_lengths_modes():
    records = [("a", b"ACGT"), ("b", b"TT")]
    padded, length = native.fasta_normalize_lengths(records, "pad")
    assert length == 4
    assert padded == [("a", b"ACGT"), ("b", b"TT--")]

    with pytest.raises(Exception, match="length 2 but expected 4"):
        native.fasta_normalize_lengths(records, "strict")


def test_parse_date_from_header():
    d = native.parse_date_from_header("seq_2021-07-15")
    # ISO dates take the calendar-precise day-of-year midpoint:
    # 2021-07-15 is day 196 of 365, so (196 - 0.5) / 365 + 2021.
    assert abs(d - 2021.5356) < 1e-3
    assert native.parse_date_from_header("seq|2021.5|x") == 2021.5
    assert math.isnan(native.parse_date_from_header("no_date_here"))


# ---- Order statistics ----------------------------------------------------

def test_median_and_percentile():
    assert native.median([3.0, 1.0, 2.0]) == 2.0
    assert math.isnan(native.median([]))
    # numpy-linear percentile on already-sorted input.
    assert native.percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
    assert math.isnan(native.percentile([], 50.0))


def test_benjamini_hochberg():
    assert native.benjamini_hochberg([0.01, 0.04, 0.5]) == pytest.approx(
        [0.03, 0.06, 0.5]
    )
    # q-values are clipped to [0, 1] and monotone non-decreasing in p.
    q = native.benjamini_hochberg([0.9, 0.95, 0.99])
    assert all(0.0 <= v <= 1.0 for v in q)


# ---- Mixture-null p-values -----------------------------------------------

def test_self_liang_mixture():
    # 0.5·δ(0) + 0.5·χ²₁. Contract (aeon-manifold/stats.rs): LRT <= 0 —
    # including NaN, since the predicate is `l > 0.0` — takes the
    # point-mass branch p = 1.0 (BUSTED convention).
    assert native.pvals_from_lrt_self_liang([0.0, -1.0, float("nan")]) == [
        1.0,
        1.0,
        1.0,
    ]
    # LRT = 3.84 -> 0.5 * sf_chi2_1(3.84) = 0.5 * 0.05004 = 0.02502.
    p = native.pvals_from_lrt_self_liang([3.84])[0]
    expected = 0.5 * math.erfc(math.sqrt(3.84 / 2.0))
    assert p == pytest.approx(expected, abs=1e-6)


def test_meme_mixture():
    # 1/3·δ(0) + 2/3·(0.45·χ²₁ + 0.55·χ²₂). Contract: LRT <= 0 takes the
    # point-mass branch p = 2/3 (the HyPhy MEME null).
    assert native.pvals_from_lrt_meme([0.0, -1.0]) == pytest.approx(
        [2.0 / 3.0, 2.0 / 3.0]
    )
    p = native.pvals_from_lrt_meme([3.84])[0]
    sf1 = math.erfc(math.sqrt(3.84 / 2.0))  # χ²₁ sf
    sf2 = math.exp(-3.84 / 2.0)             # χ²₂ sf
    expected = (2.0 / 3.0) * (0.45 * sf1 + 0.55 * sf2)
    assert p == pytest.approx(expected, abs=1e-6)


def test_cauchy_combination():
    assert native.cauchy_combination_p([]) == 1.0
    p_small = native.cauchy_combination_p([0.001, 0.001])
    p_mid = native.cauchy_combination_p([0.5, 0.5])
    assert 0.0 < p_small < p_mid < 1.0


# ---- AutoClock bindings ---------------------------------------------------

def _additive_clock(n=10, mu=0.004, t0=2020.0, span=4.0):
    dates = [t0 + span * i / (n - 1) for i in range(n)]
    d = [mu * abs(dates[i] - dates[j]) for i in range(n) for j in range(n)]
    return dates, d


def test_fit_clock_community_json_contract():
    n = 10
    dates, d = _additive_clock(n)
    rec = json.loads(native.fit_clock_community(list(range(n)), dates, d, n))
    # Perfectly additive clock: the fit must recover mu and t0 exactly.
    assert rec["fieller_status"] == "OK"
    assert rec["mu"] == pytest.approx(0.004, abs=1e-9)
    assert rec["tmrca"] == pytest.approx(2020.0, abs=1e-9)
    assert rec["size"] == n
    assert rec["members"] == list(range(n))
    # chronaeon reshapes this JSON into its fit-dict; pin the keys it
    # reads so a Rust-side rename fails here, not downstream.
    for key in (
        "aicc", "ci_mrca", "ci_mu", "d0", "n_eff", "operational_tier",
        "r_squared", "rss", "se_mu", "stop_reason", "time_span_years",
        "tot_var", "is_active_outbreak", "path", "fieller_status",
    ):
        assert key in rec, f"missing key {key!r}"


def test_recursive_spectral_autoclock_splits_clusters():
    nh, n = 8, 16
    dates = [2020.0 + 0.1 * (i % nh) for i in range(n)]
    d = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i * n + j] = 0.002 if (i < nh) == (j < nh) else 0.4
    leaves = json.loads(
        native.recursive_spectral_autoclock(list(range(n)), dates, d, n)
    )
    assert len(leaves) == 2
    paths = sorted(leaf["path"] for leaf in leaves)
    assert paths == ["root.0", "root.1"]
    members = sorted(m for leaf in leaves for m in leaf["members"])
    assert members == list(range(n))
    # Member lists arrive in Fiedler order, not sorted order.
    planted = sorted(sorted(leaf["members"]) for leaf in leaves)
    assert planted == [list(range(nh)), list(range(nh, n))]
