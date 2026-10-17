"""
test_autoclock.py
-----------------
Dedicated tests for chronaeon.autoclock covering:
- AutoClockDeconvolution with synthetic multi-community data
- run_autoclock_deconvolution end-to-end
- fit_fast_ols_clock
- HierarchicalAutoClock basic construction
- select_adaptive_n_landmarks edge cases
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronaeon.autoclock import (
    AutoClockDeconvolution,
    HierarchicalAutoClock,
    fit_clock,
    recursive_spectral_autoclock,
    run_autoclock_deconvolution,
    run_hierarchical_autoclock,
    select_adaptive_n_landmarks,
    fit_fast_ols_clock,
)


def _make_synthetic_two_clade_fasta(tmp_path, n_per_clade=12, seq_len=60):
    """Create a synthetic FASTA with two distinct clades having different mutation rates."""
    clade1_taxa = [f"c1_t{i}" for i in range(1, n_per_clade + 1)]
    clade2_taxa = [f"c2_t{i}" for i in range(1, n_per_clade + 1)]
    all_taxa = clade1_taxa + clade2_taxa

    dates_map = {}
    seqs = {}
    for i, t in enumerate(clade1_taxa):
        dates_map[t] = 1990.0 + i * 2.0
        seqs[t] = "AAAA" + "A" * (seq_len - 4 - i) + "C" * i + "T" * 4
    for i, t in enumerate(clade2_taxa):
        dates_map[t] = 2000.0 + i * 1.5
        seqs[t] = "GGGG" + "G" * (seq_len - 4 - i) + "T" * i + "C" * 4

    fasta_path = tmp_path / "autoclock_test.fasta"
    with open(fasta_path, "w") as f:
        for t in all_taxa:
            f.write(f">{t}\n{seqs[t]}\n")

    dates_csv = tmp_path / "autoclock_dates.csv"
    pd.DataFrame({"id": all_taxa, "date": [dates_map[t] for t in all_taxa]}).to_csv(dates_csv, index=False)

    return fasta_path, dates_csv, all_taxa


class TestSelectAdaptiveNLandmarks:
    def test_small_dataset_clamped_to_n_taxa(self):
        m = select_adaptive_n_landmarks(n_taxa=50, max_k=4)
        assert m == 50

    def test_medium_dataset_sublinear(self):
        m = select_adaptive_n_landmarks(n_taxa=2873, max_k=8, max_memory_mb=1024.0)
        assert 500 <= m <= 2500

    def test_huge_dataset_memory_bound(self):
        m = select_adaptive_n_landmarks(n_taxa=100000, max_k=8, max_memory_mb=100.0)
        assert m == 125

    def test_user_override(self):
        m = select_adaptive_n_landmarks(n_taxa=5000, user_landmarks=500)
        assert m == 500

    def test_user_override_string_none(self):
        m = select_adaptive_n_landmarks(n_taxa=5000, user_landmarks="none")
        assert m is not None
        assert m > 0


class TestFitFastOLSClock:
    def test_linear_recovery(self):
        """fit_fast_ols_clock should recover known slope and intercept."""
        np.random.seed(42)
        true_t0 = 1950.0
        true_mu = 0.001
        times = np.linspace(1970, 2020, 50)
        noise = np.random.normal(0, 0.0005, size=len(times))
        dists = true_mu * (times - true_t0) + noise

        result = fit_fast_ols_clock(times, dists)
        assert "mu" in result
        assert "t_mrca" in result
        assert "r2" in result
        assert abs(result["mu"] - true_mu) < 0.0002
        assert abs(result["t_mrca"] - true_t0) < 10.0

    def test_zero_distance(self):
        """All-zero distances should produce mu=0."""
        times = np.linspace(2000, 2020, 10)
        dists = np.zeros(10)
        result = fit_fast_ols_clock(times, dists)
        assert result["mu"] == 0.0


class TestAutoClockDeconvolution:
    def test_run_autoclock_two_clades(self, tmp_path):
        """run_autoclock_deconvolution should detect 2 communities in synthetic data."""
        fasta_path, dates_csv, all_taxa = _make_synthetic_two_clade_fasta(tmp_path)

        result = run_autoclock_deconvolution(
            alignment_path=fasta_path,
            dates_source=dates_csv,
            max_k=3,
            manifold="tn93",
            quiet=True,
            output_dir=tmp_path / "out",
        )

        assert result["optimal_k"] >= 1
        assert len(result["communities"]) >= 1
        assert "classified_metadata_path" in result
        assert Path(result["classified_metadata_path"]).exists()

    def test_run_autoclock_with_dates_csv(self, tmp_path):
        """run_autoclock_deconvolution should accept dates_source as CSV."""
        fasta_path, dates_csv, _ = _make_synthetic_two_clade_fasta(tmp_path)

        result = run_autoclock_deconvolution(
            alignment_path=fasta_path,
            dates_source=dates_csv,
            max_k=2,
            manifold="tn93",
            quiet=True,
            output_dir=tmp_path / "out2",
        )

        assert result["optimal_k"] >= 1

    def test_autoclock_output_dir_created(self, tmp_path):
        """Output directory should be created if it doesn't exist."""
        fasta_path, dates_csv, _ = _make_synthetic_two_clade_fasta(tmp_path)

        out_dir = tmp_path / "nonexistent" / "nested"
        result = run_autoclock_deconvolution(
            alignment_path=fasta_path,
            dates_source=dates_csv,
            max_k=2,
            manifold="tn93",
            quiet=True,
            output_dir=out_dir,
        )

        assert out_dir.exists()


class TestHierarchicalAutoClock:
    def test_hierarchical_basic(self, tmp_path):
        """run_hierarchical_autoclock should produce a valid tree and community assignments."""
        fasta_path, dates_csv, all_taxa = _make_synthetic_two_clade_fasta(tmp_path, n_per_clade=10)

        result = run_hierarchical_autoclock(
            alignment_path=fasta_path,
            dates_source=dates_csv,
            max_depth=2,
            min_leaf_size=5,
            min_delta_aicc=5.0,
            manifold="tn93",
            output_dir=tmp_path / "hier_out",
            quiet=True,
        )

        assert result["total_taxa"] == 20
        assert result["n_leaves"] >= 1
        assert "leaf_communities" in result
        assert Path(result["tree_path"]).exists()
        assert Path(result["classified_metadata_path"]).exists()

    def test_hierarchical_no_split(self, tmp_path):
        """With high min_delta_aicc, should produce a single community (no split)."""
        fasta_path, dates_csv, _ = _make_synthetic_two_clade_fasta(tmp_path, n_per_clade=10)

        result = run_hierarchical_autoclock(
            alignment_path=fasta_path,
            dates_source=dates_csv,
            max_depth=1,
            min_leaf_size=20,
            min_delta_aicc=1000.0,
            manifold="tn93",
            output_dir=tmp_path / "hier_nosplit",
            quiet=True,
        )

        assert result["n_leaves"] == 1


class TestNativeReshapeContract:
    """``fit_clock`` / ``recursive_spectral_autoclock`` delegate to
    ``aeon_core._native`` and reshape the serialized SteveCommunity JSON
    into the chronaeon fit-dict. These tests pin that contract so a
    Rust-side field rename fails here rather than silently producing
    missing keys downstream. (The legacy pure-Python kernel was removed;
    chronaeon hard-depends on the native extension, so there is no
    fallback path to parity-test.)"""

    FIT_KEYS = {
        "n", "mu", "se", "r2", "rss", "tot_var", "aicc", "tmrca",
        "ci_mrca", "fieller_status", "mean_dist", "max_dist", "span",
        "d0", "n_eff", "ci_mu", "is_active_outbreak", "operational_tier",
    }

    def _additive_clock(self, n=10, mu=0.004, t0=2020.0, span=4.0):
        dates = np.array([t0 + span * i / (n - 1) for i in range(n)])
        D = np.abs(dates[:, None] - dates[None, :]) * mu
        return dates, D

    def _two_clusters(self, nh=8):
        n = 2 * nh
        dates = np.array([2020.0 + 0.1 * (i % nh) for i in range(n)])
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = 0.002 if (i < nh) == (j < nh) else 0.4
        return dates, D

    def test_fit_clock_returns_full_fit_dict(self):
        n = 10
        dates, D = self._additive_clock(n)
        fit = fit_clock(np.arange(n), dates, D)
        assert self.FIT_KEYS <= set(fit.keys()), (
            f"missing keys: {self.FIT_KEYS - set(fit.keys())}"
        )
        # Perfectly additive clock: exact recovery of mu and t0.
        assert fit["mu"] == pytest.approx(0.004, abs=1e-9)
        assert fit["tmrca"] == pytest.approx(2020.0, abs=1e-9)
        assert fit["n"] == n

    def test_recursive_autoclock_tuple_contract(self):
        nh, n = 8, 16
        dates, D = self._two_clusters(nh)
        leaves = recursive_spectral_autoclock(np.arange(n), dates, D)
        assert len(leaves) == 2
        for sub_idx, fit, stop_reason, path in leaves:
            assert isinstance(sub_idx, np.ndarray)
            assert self.FIT_KEYS <= set(fit.keys())
            assert isinstance(stop_reason, str) and stop_reason
            assert isinstance(path, str) and path.startswith("root.")
        members = sorted(int(m) for leaf in leaves for m in leaf[0])
        assert members == list(range(n))

    def test_is_ref_is_currently_inert(self):
        # `is_ref` is threaded through for signature parity but documented
        # as inert on both sides (reserved for reference-anchor
        # weighting). If it ever becomes active this test should fail
        # loudly rather than silently change partitioning.
        nh, n = 8, 16
        dates, D = self._two_clusters(nh)
        is_ref = np.zeros(n, dtype=bool)
        is_ref[:4] = True
        plain = recursive_spectral_autoclock(np.arange(n), dates, D)
        with_ref = recursive_spectral_autoclock(
            np.arange(n), dates, D, is_ref_array=is_ref
        )
        assert len(plain) == len(with_ref)
        for (idx_a, fit_a, stop_a, path_a), (idx_b, fit_b, stop_b, path_b) in zip(
            plain, with_ref
        ):
            np.testing.assert_array_equal(idx_a, idx_b)
            assert stop_a == stop_b and path_a == path_b
            assert fit_a == fit_b
