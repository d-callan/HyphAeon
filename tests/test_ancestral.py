"""Unit tests for pure functions in hyphaeon.ancestral + CLI smoke test."""
import argparse
import numpy as np
import pytest

from hyphaeon.ancestral import (
    AncestralCandidate,
    compute_centroid_tip_weights,
    compute_base_logprobs,
    compute_site_confidence,
    _dms_regulariser,
    build_pairwise_cooccurrence,
    _score_sequence,
    _greedy_optimise,
    _build_parser,
    DEFAULT_LAMBDA_DIST,
    DEFAULT_LAMBDA_PLAST,
    DEFAULT_LAMBDA_EPI,
    DEFAULT_PSEUDOCOUNT,
)


class TestComputeCentroidTipWeights:
    def test_normalised(self):
        z = np.random.randn(5, 4).astype(np.float32)
        w = compute_centroid_tip_weights(z, DEFAULT_LAMBDA_DIST)
        assert w.shape == (5,)
        assert np.isclose(w.sum(), 1.0)

    def test_origin_anchor_matches_default(self):
        z = np.random.randn(4, 4).astype(np.float32)
        w_default = compute_centroid_tip_weights(z, 3.0)
        w_origin = compute_centroid_tip_weights(z, 3.0, anchor=np.zeros(4))
        assert np.allclose(w_default, w_origin)

    def test_outgroup_anchor_highest_weight(self):
        z = np.array([[1.0, 0, 0, 0],
                       [2.0, 0, 0, 0],
                       [0.5, 0, 0, 0]], dtype=np.float32)
        anchor = z[2].copy()
        w = compute_centroid_tip_weights(z, 4.0, anchor=anchor)
        assert w.shape == (3,)
        assert np.isclose(w.sum(), 1.0)
        assert np.argmax(w) == 2

    def test_all_zero_coords_uniform(self):
        z = np.zeros((3, 4), dtype=np.float32)
        w = compute_centroid_tip_weights(z, 4.0)
        assert np.allclose(w, 1.0 / 3.0)


class TestComputeBaseLogprobs:
    def test_shape_and_normalisation(self):
        a_np = np.array([[0, 1, 0], [2, 2, 20]], dtype=np.int64)
        w = np.array([0.5, 0.3, 0.2])
        logp = compute_base_logprobs(a_np, w, DEFAULT_PSEUDOCOUNT)
        assert logp.shape == (2, 20)
        probs = np.exp(logp)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_gap_ignored(self):
        a_np = np.array([[5, 5, 20]], dtype=np.int64)
        w = np.array([0.4, 0.4, 0.2])
        logp = compute_base_logprobs(a_np, w, DEFAULT_PSEUDOCOUNT)
        probs = np.exp(logp)
        assert probs[0, 5] > probs[0, 0]  # observed AA gets more mass
        assert np.isclose(probs.sum(), 1.0)

    def test_pseudocount_smooths(self):
        a_np = np.array([[0, 0, 0]], dtype=np.int64)
        w = np.array([1.0, 1.0, 1.0])
        logp = compute_base_logprobs(a_np, w, 0.5)
        probs = np.exp(logp)
        assert probs[0, 0] > probs[0, 1]  # observed AA dominates
        assert probs[0, 1] > 0  # pseudocount gives nonzero mass


class TestComputeSiteConfidence:
    def test_zero_lrt_gives_one(self):
        lrts = np.array([0.0, 0.0])
        conf = compute_site_confidence(lrts)
        assert np.allclose(conf, 1.0)

    def test_high_lrt_gives_low_confidence(self):
        lrts = np.array([0.0, 100.0])
        conf = compute_site_confidence(lrts)
        assert conf[0] > conf[1]
        assert conf[1] < 0.02

    def test_clipped_negative(self):
        lrts = np.array([-5.0])
        conf = compute_site_confidence(lrts)
        assert np.isclose(conf[0], 1.0)

    def test_monotonic_decreasing(self):
        lrts = np.array([0.0, 1.0, 5.0, 20.0, 100.0])
        conf = compute_site_confidence(lrts)
        assert np.all(np.diff(conf) < 0)


class TestDmsRegulariser:
    def test_positive_delta_penalised(self):
        deltas = np.array([[5.0, -3.0, 0.0]])
        pen = _dms_regulariser(deltas)
        assert pen[0, 0] == -5.0
        assert pen[0, 1] == 0.0
        assert pen[0, 2] == 0.0

    def test_all_zero(self):
        deltas = np.zeros((3, 20))
        pen = _dms_regulariser(deltas)
        assert np.all(pen == 0.0)


class TestBuildPairwiseCooccurrence:
    def test_shape_and_keys(self):
        a_np = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        edges = [{"site_u": 1, "site_v": 2}]
        tables = build_pairwise_cooccurrence(a_np, edges, 1e-3)
        assert (0, 1) in tables
        assert tables[(0, 1)].shape == (20, 20)

    def test_log_freq_sums_to_one(self):
        a_np = np.array([[0, 1], [2, 3]], dtype=np.int64)
        edges = [{"site_u": 1, "site_v": 2}]
        tables = build_pairwise_cooccurrence(a_np, edges, 0.1)
        probs = np.exp(tables[(0, 1)])
        assert np.isclose(probs.sum(), 1.0)

    def test_observed_pair_dominates(self):
        a_np = np.array([[0, 0, 0], [5, 5, 5]], dtype=np.int64)
        edges = [{"site_u": 1, "site_v": 2}]
        tables = build_pairwise_cooccurrence(a_np, edges, 1e-3)
        table = tables[(0, 1)]
        flat_idx = np.argmax(table)
        assert divmod(flat_idx, 20) == (0, 5)

    def test_empty_edges(self):
        a_np = np.zeros((3, 3), dtype=np.int64)
        tables = build_pairwise_cooccurrence(a_np, [], 1e-3)
        assert len(tables) == 0


class TestScoreSequence:
    def test_consistency(self):
        tokens = np.array([0, 5, 10])
        base_logp = np.random.randn(3, 20)
        confidence = np.array([0.9, 0.8, 0.7])
        dms_penalty = np.zeros((3, 20))
        cooccur = {}
        total, base, dms, epi = _score_sequence(
            tokens, base_logp, confidence, dms_penalty, cooccur,
            DEFAULT_LAMBDA_PLAST, DEFAULT_LAMBDA_EPI,
        )
        expected_base = float((confidence * base_logp[np.arange(3), tokens]).sum())
        assert np.isclose(base, expected_base)
        assert np.isclose(dms, 0.0)
        assert np.isclose(epi, 0.0)
        assert np.isclose(total, base)

    def test_with_epistasis(self):
        tokens = np.array([0, 5])
        base_logp = np.zeros((2, 20))
        confidence = np.ones(2)
        dms_penalty = np.zeros((2, 20))
        cooccur = {(0, 1): np.log(np.full((20, 20), 1.0 / 400))}
        total, base, dms, epi = _score_sequence(
            tokens, base_logp, confidence, dms_penalty, cooccur,
            0.0, 0.5,
        )
        expected_epi = 0.5 * np.log(1.0 / 400)
        assert np.isclose(epi, expected_epi)


class TestGreedyOptimise:
    def test_converges_on_consensus(self):
        L, N = 10, 5
        a_np = np.zeros((L, N), dtype=np.int64)
        a_np[:, 1] = 3  # one taxon has AA 3 everywhere
        w = np.full(N, 1.0 / N)
        base_logp = compute_base_logprobs(a_np, w, 1e-3)
        confidence = np.ones(L)
        dms_penalty = np.zeros((L, 20))
        cooccur = {}
        init = np.full(L, 3)
        result = _greedy_optimise(init, base_logp, confidence, dms_penalty, cooccur, 0.15, 0.5)
        assert np.all(result == 0)  # consensus (AA 0) should win

    def test_no_change_when_optimal(self):
        L = 5
        base_logp = np.zeros((L, 20))
        base_logp[:, 0] = 10.0  # AA 0 is overwhelmingly best
        confidence = np.ones(L)
        dms_penalty = np.zeros((L, 20))
        cooccur = {}
        init = np.zeros(L, dtype=np.int64)
        result = _greedy_optimise(init, base_logp, confidence, dms_penalty, cooccur, 0.15, 0.5)
        assert np.array_equal(result, init)


class TestAncestralCandidate:
    def test_to_dict_roundtrip(self):
        c = AncestralCandidate(
            sequence="ACDEFGHIKL",
            total_score=-5.0,
            base_score=-3.0,
            dms_score=-1.0,
            epi_score=-1.0,
        )
        d = c.to_dict()
        assert d["sequence"] == "ACDEFGHIKL"
        assert d["total_score"] == -5.0
        assert d["base_score"] == -3.0
        assert d["dms_score"] == -1.0
        assert d["epi_score"] == -1.0


class TestCLIParser:
    def test_build_parser_outgroup(self):
        parser = _build_parser()
        args = parser.parse_args(["-a", "test.fasta", "--outgroup", "Outgroup1"])
        assert args.outgroup == "Outgroup1"

    def test_build_parser_no_outgroup(self):
        parser = _build_parser()
        args = parser.parse_args(["-a", "test.fasta"])
        assert args.outgroup is None

    def test_build_parser_all_args(self):
        parser = _build_parser()
        args = parser.parse_args([
            "-a", "aln.fa", "-t", "tree.nwk", "--outgroup", "OG",
            "-k", "3", "--lambda-dist", "2.0", "--sweeps", "5",
            "--seed", "42", "--cpu",
        ])
        assert args.alignment == "aln.fa"
        assert args.tree == "tree.nwk"
        assert args.outgroup == "OG"
        assert args.top_k == 3
        assert args.lambda_dist == 2.0
        assert args.sweeps == 5
        assert args.seed == 42
        assert args.cpu is True

    def test_cli_subcommand_registered(self):
        from hyphaeon.cli import main as cli_main
        cli_parser = argparse.ArgumentParser()
        sub = cli_parser.add_subparsers(dest="command")
        sub.add_parser("ancestral", aliases=["asr"])
        args = cli_parser.parse_args(["ancestral"])
        assert args.command == "ancestral"
        args = cli_parser.parse_args(["asr"])
        assert args.command == "asr"
