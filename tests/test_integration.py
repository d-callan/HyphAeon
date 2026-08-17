"""Integration tests for the inference CLI plumbing.

These tests exercise the data pipeline and CLI mechanics — not model
accuracy. They use a tiny dummy checkpoint (random weights, minimal
architecture) instead of real pretrained weights, so they don't depend on
the 23 MB weights file being present. Real weights will eventually move to
Hugging Face / GitHub releases and new model versions will ship different
weights; package tests should not be coupled to any of that.

`axomeme_lrt` and `p_value` are model predictions that fluctuate across
torch versions, hardware, and model weights, so we only check that the
pipeline runs end-to-end and produces well-formed, finite, non-trivial
output. The deterministic columns (`site`, `is_invariable`) are checked
exactly since they depend only on the input alignment, not the model.

Model accuracy / regression testing belongs in a separate training-focused
test suite and is out of scope here.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import EXPECTED_DIR

EXPECTED_COLUMNS = ["site", "axomeme_lrt", "p_value", "is_invariable"]


def run_cli(fasta, tree, weights, csv_out, cpu=True):
    cmd = [
        sys.executable, "-m", "axomeme.cli",
        "predict",
        "-a", fasta,
        "-t", tree,
        "-w", weights,
        "-c", csv_out,
    ]
    if cpu:
        cmd.append("--cpu")
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.parametrize("name,fasta,tree,csv", [
    ("Smc6", "Smc6.fasta", "Smc6.nwk", "Smc6_results.csv"),
    ("bat_oas1", "bat_oas1.fasta", "bat_oas1.nwk", "bat_oas1_results.csv"),
    ("camelid", "camelid.fasta", "camelid.nwk", "camelid_results.csv"),
])
class TestExampleDatasets:
    def test_cli_runs_and_csv_is_well_formed(self, name, fasta, tree, csv, examples_dir, dummy_weights, tmp_path):
        fa = os.path.join(examples_dir, fasta)
        nwk = os.path.join(examples_dir, tree)
        expected = os.path.join(EXPECTED_DIR, csv)
        out = str(tmp_path / "output.csv")

        result = run_cli(fa, nwk, dummy_weights, out)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(out), "CSV output not created"

        actual = pd.read_csv(out)
        expected_df = pd.read_csv(expected)

        # Structure: same columns, same row count.
        assert list(actual.columns) == EXPECTED_COLUMNS, f"Columns mismatch: {actual.columns}"
        assert len(actual) == len(expected_df), f"Row count mismatch: {len(actual)} vs {len(expected_df)}"

        # Deterministic columns (data properties, not model outputs): exact match.
        assert (actual["site"] == expected_df["site"]).all(), "site column mismatch"
        assert (actual["is_invariable"] == expected_df["is_invariable"]).all(), "is_invariable column mismatch"

        # Model output columns: finite (no NaN/inf), in valid range, and not
        # all-zero (smoke check that the model actually produced predictions).
        for col in ("axomeme_lrt", "p_value"):
            assert np.isfinite(actual[col]).all(), f"{col} contains non-finite values"
        assert (actual["axomeme_lrt"] >= 0).all(), "axomeme_lrt should be non-negative"
        assert ((actual["p_value"] >= 0) & (actual["p_value"] <= 1)).all(), "p_value out of [0, 1]"
        assert (actual["axomeme_lrt"] > 0).any(), "axomeme_lrt is all zero — model did not produce predictions"

    def test_json_output(self, name, fasta, tree, csv, examples_dir, dummy_weights, tmp_path):
        fa = os.path.join(examples_dir, fasta)
        nwk = os.path.join(examples_dir, tree)
        out_json = str(tmp_path / "output.json")
        out_csv = str(tmp_path / "output.csv")
        cmd = [
            sys.executable, "-m", "axomeme.cli",
            "predict",
            "-a", fa,
            "-t", nwk,
            "-w", dummy_weights,
            "-o", out_json,
            "-c", out_csv,
            "--cpu",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(out_json), "JSON output not created"
        with open(out_json) as f:
            data = json.load(f)
        assert "taxa_count" in data
        assert "codon_count" in data
        assert "sites" in data
        assert len(data["sites"]) == data["codon_count"]


def test_batch_size_one_produces_same_structure(examples_dir, dummy_weights, tmp_path):
    """Site batching (chunked inference) must produce the same output structure
    as the default path. Runs with --batch-size 1 (extreme chunking, one site
    per forward pass) and checks the CSV is well-formed with the right row
    count, columns, and finite values.
    """
    fa = os.path.join(examples_dir, "Smc6.fasta")
    nwk = os.path.join(examples_dir, "Smc6.nwk")
    expected = os.path.join(EXPECTED_DIR, "Smc6_results.csv")
    out = str(tmp_path / "batched.csv")

    cmd = [
        sys.executable, "-m", "axomeme.cli", "predict",
        "-a", fa, "-t", nwk, "-w", dummy_weights, "-c", out,
        "--cpu", "--batch-size", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    actual = pd.read_csv(out)
    expected_df = pd.read_csv(expected)
    assert list(actual.columns) == EXPECTED_COLUMNS
    assert len(actual) == len(expected_df)
    assert (actual["site"] == expected_df["site"]).all()
    assert (actual["is_invariable"] == expected_df["is_invariable"]).all()
    for col in ("axomeme_lrt", "p_value"):
        assert np.isfinite(actual[col]).all(), f"{col} contains non-finite values"
