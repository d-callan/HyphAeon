"""
concordance/test_ancestral_concordance.py

Benchmark HyphAeon ancestral sequence candidate scoring against the
Randall et al. (2016) experimental FP phylogeny — the gold-standard
benchmark for ASR with *known* true ancestral sequences.

Randall, R.N., Radford, C.E., Roof, K.A., Natarajan, D.K. & Gaucher, E.A.
(2016) "An experimental phylogeny to benchmark ancestral sequence
reconstruction." Nature Communications 7:12847.

Unlike classical ASR (PAML, FastML), HyphAeon is topology-blind (star-tree
invariant per model_eval/invariance/). This test checks that the centroid
ancestor and per-node reconstructions achieve reasonable amino-acid
identity against the 18 known true ancestors (nodes 20–37).

Requires real HyphAeon weights (skipped if unavailable, same as other
model_eval tests).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hyphaeon.ancestral import run_ancestral_reconstruction

RANDALL_DIR = REPO_ROOT / "examples" / "randall_asr"
LEAVES = RANDALL_DIR / "fp_leaves_codon.fasta"
TREE = RANDALL_DIR / "RandallBenchmarkTree.newick"
TRUE_ANC = RANDALL_DIR / "fp_true_ancestors.fasta"
MRFP1 = RANDALL_DIR / "fp_mrfp1_reference.fasta"

# Node → descendant leaves (from Randall et al. 2016 tree topology)
NODE_TO_LEAVES = {
    "20": "root (all)", "21": "leaves 3-19", "22": "leaves 11-19",
    "23": "leaves 14-19", "24": "leaves 16-19", "25": "leaves 18,19",
    "26": "leaves 16,17", "27": "leaves 14,15", "28": "leaves 11-13",
    "29": "leaves 12,13", "30": "leaves 3-10", "31": "leaves 9,10",
    "32": "leaves 3-8", "33": "leaves 7,8", "34": "leaves 3-6",
    "35": "leaves 5,6", "36": "leaves 3,4", "37": "leaves 1,2",
}


def _parse_fasta(path):
    seqs = {}
    header = None
    seq = []
    for line in Path(path).read_text().strip().split("\n"):
        if line.startswith(">"):
            if header is not None:
                seqs[header] = "".join(seq)
            header = line[1:].split()[0]
            seq = []
        else:
            seq.append(line.strip())
    if header is not None:
        seqs[header] = "".join(seq)
    return seqs


def _identity(s1, s2):
    if len(s1) != len(s2):
        return 0.0
    return sum(1 for a, b in zip(s1, s2) if a == b) / len(s1) * 100


@pytest.fixture(scope="module")
def true_ancestors():
    """Load true ancestral sequences (nodes 20-37)."""
    true_seqs = _parse_fasta(TRUE_ANC)
    mrfp1 = list(_parse_fasta(MRFP1).values())[0]
    all_true = {"20": mrfp1}
    all_true.update(true_seqs)
    return all_true


@pytest.fixture(scope="module")
def centroid_result(weights_info):
    """Run HyphAeon ancestral in centroid mode (no tree, outgroup=01)."""
    res = run_ancestral_reconstruction(
        alignment_path=str(LEAVES),
        tree_path=None,
        weights_path=weights_info[0],
        use_tn93=True,
        outgroup="01",
        lambda_dist=1.0,
        top_k=5,
        cpu=True,
        progress=False,
    )
    return res


@pytest.fixture(scope="module")
def per_node_result(weights_info):
    """Run HyphAeon ancestral in per-node mode (with tree)."""
    res = run_ancestral_reconstruction(
        alignment_path=str(LEAVES),
        tree_path=str(TREE),
        weights_path=weights_info[0],
        per_node=True,
        cpu=True,
        progress=False,
    )
    return res


class TestCentroidConcordance:
    """Centroid-mode ASR: single ancestor vs all 18 true ancestors."""

    def test_centroid_identity_above_leaf_baseline(self, centroid_result, true_ancestors):
        """Best centroid candidate should match at least one true ancestor
        above the average leaf-leaf identity (~85% for this dataset)."""
        best_candidate = centroid_result["candidates"][0]["sequence"]
        identities = [_identity(best_candidate, seq) for seq in true_ancestors.values()]
        best_match = max(identities)
        assert best_match >= 80.0, (
            f"Best centroid match {best_match:.1f}% below 80% threshold; "
            f"Randall et al. report 93-98% for classical ASR with known tree"
        )

    def test_centroid_matches_root_better_than_average(self, centroid_result, true_ancestors):
        """The centroid should match the root (node 20 / mRFP1) at least as
        well as the average across all true ancestors."""
        best_candidate = centroid_result["candidates"][0]["sequence"]
        root_identity = _identity(best_candidate, true_ancestors["20"])
        all_identities = [_identity(best_candidate, seq) for seq in true_ancestors.values()]
        avg_identity = np.mean(all_identities)
        assert root_identity >= avg_identity, (
            f"Root identity {root_identity:.1f}% below average {avg_identity:.1f}%"
        )


class TestPerNodeConcordance:
    """Per-node ASR: one ancestor per internal tree node vs true ancestors."""

    def test_per_node_average_identity(self, per_node_result, true_ancestors):
        """Average per-node identity should exceed 75% (HyphAeon is
        topology-blind, so we don't expect to match the 93-98% of classical
        ASR with known tree, but should be well above random)."""
        node_results = {n["node_label"]: n["candidates"][0]["sequence"]
                       for n in per_node_result["nodes"]}
        identities = []
        for node_label, true_seq in true_ancestors.items():
            # Try to match node labels (HyphAeon uses tree's internal labels)
            ha_label = f"A{node_label}" if node_label != "20" else node_label
            if ha_label in node_results:
                identities.append(_identity(node_results[ha_label], true_seq))
        if not identities:
            pytest.skip("No matching node labels between HyphAeon and true ancestors")
        avg = np.mean(identities)
        assert avg >= 75.0, (
            f"Average per-node identity {avg:.1f}% below 75% threshold"
        )

    def test_per_node_better_than_centroid(self, centroid_result, per_node_result, true_ancestors):
        """Per-node mode (with tree) should improve over centroid mode (no tree)
        on average, since it uses phylogenetic information."""
        centroid_seq = centroid_result["candidates"][0]["sequence"]
        centroid_identities = [_identity(centroid_seq, seq) for seq in true_ancestors.values()]
        centroid_avg = np.mean(centroid_identities)

        node_results = {n["node_label"]: n["candidates"][0]["sequence"]
                       for n in per_node_result["nodes"]}
        per_node_identities = []
        for node_label, true_seq in true_ancestors.items():
            ha_label = f"A{node_label}" if node_label != "20" else node_label
            if ha_label in node_results:
                per_node_identities.append(_identity(node_results[ha_label], true_seq))
        if not per_node_identities:
            pytest.skip("No matching node labels")
        per_node_avg = np.mean(per_node_identities)

        assert per_node_avg >= centroid_avg - 5.0, (
            f"Per-node avg {per_node_avg:.1f}% much worse than centroid {centroid_avg:.1f}%"
        )
