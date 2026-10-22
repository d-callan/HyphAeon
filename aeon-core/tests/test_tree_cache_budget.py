"""
Tests for the --max-memory-mb tree attention bias budget gate.

The all-layers Markov bias cache costs
num_layers * num_heads * (N+1)^2 * 4 bytes (~28.8 GB at N=10,000 for the
production 6-layer/12-head model). When the estimate exceeds the user's
budget, precompute_tree_cache keeps only the (N+1)^2 distance tensor under
"static_dist" and consumers evaluate per-layer biases lazily via
tree_cache_layer_bias. Numerical outputs must be identical either way.
"""

import numpy as np
import torch

from aeon_core.model import (
    DEFAULT_MAX_MEMORY_MB,
    PhyloAxialTransformer,
    tree_cache_layer_bias,
)


def _tree_inputs(num_species=6, seed=0):
    rng = np.random.default_rng(seed)
    dist = rng.uniform(0.01, 0.5, size=(num_species, num_species))
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    mds = rng.normal(0, 1, size=(num_species, 4))
    return dist, mds


def _model():
    torch.manual_seed(0)
    model = PhyloAxialTransformer(
        embed_dim=16, num_layers=2, num_heads=2, window_size=1
    )
    model.eval()
    return model


def _site_batch(num_species=6, num_codons=12, seed=42):
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(0, 64, (num_codons, num_species, 1), generator=g)
    a = torch.randint(0, 20, (num_codons, num_species, 1), generator=g)
    return c, a


class TestBudgetGate:
    def test_static_biases_under_budget(self):
        model = _model()
        dist, mds = _tree_inputs()
        cache = model.precompute_tree_cache(dist, mds)
        assert isinstance(cache["static_phylo_biases"], list)
        assert len(cache["static_phylo_biases"]) == len(model.row_layers)
        # The distance tensor is always cached for the lazy path
        assert cache["static_dist"].shape[-1] == len(dist) + 1

    def test_lazy_biases_over_budget(self):
        model = _model()
        dist, mds = _tree_inputs()
        cache = model.precompute_tree_cache(dist, mds, max_memory_mb=0.0)
        assert cache["static_phylo_biases"] is None
        assert cache["static_dist"].shape[-1] == len(dist) + 1

    def test_budget_boundary(self):
        model = _model()
        dist, mds = _tree_inputs()
        n_nodes = len(dist) + 1
        num_heads = model.row_layers[0].num_heads
        required_mb = len(model.row_layers) * num_heads * n_nodes * n_nodes * 4 / 1e6
        cache = model.precompute_tree_cache(dist, mds, max_memory_mb=required_mb * 1.01)
        assert cache["static_phylo_biases"] is not None
        cache = model.precompute_tree_cache(dist, mds, max_memory_mb=required_mb * 0.99)
        assert cache["static_phylo_biases"] is None

    def test_lazy_path_warns(self, capsys):
        model = _model()
        dist, mds = _tree_inputs()
        model.precompute_tree_cache(dist, mds, max_memory_mb=0.0)
        assert "lazy" in capsys.readouterr().out.lower()

    def test_default_budget_constant(self):
        assert DEFAULT_MAX_MEMORY_MB > 0


class TestLazyParity:
    """Static vs lazy bias evaluation must produce identical outputs."""

    def _forward_parity(self, return_attentions):
        model = _model()
        dist, mds = _tree_inputs(seed=1)
        static_cache = model.precompute_tree_cache(dist, mds)
        lazy_cache = model.precompute_tree_cache(dist, mds, max_memory_mb=0.0)
        c, a = _site_batch()
        with torch.no_grad():
            out_static = model.forward_cached(c, a, static_cache,
                                              return_attentions=return_attentions)
            out_lazy = model.forward_cached(c, a, lazy_cache,
                                            return_attentions=return_attentions)
        assert len(out_static) == len(out_lazy)
        for xs, xl in zip(out_static, out_lazy):
            if isinstance(xs, torch.Tensor):
                torch.testing.assert_close(xs, xl)
            else:  # layer_attns list
                for ts, tl in zip(xs, xl):
                    torch.testing.assert_close(ts, tl)

    def test_forward_parity(self):
        self._forward_parity(return_attentions=False)

    def test_forward_parity_return_attentions(self):
        self._forward_parity(return_attentions=True)

    def test_layer_bias_matches_static(self):
        model = _model()
        dist, mds = _tree_inputs()
        static_cache = model.precompute_tree_cache(dist, mds)
        lazy_cache = model.precompute_tree_cache(dist, mds, max_memory_mb=0.0)
        for i, layer in enumerate(model.row_layers):
            torch.testing.assert_close(
                tree_cache_layer_bias(static_cache, layer, i),
                tree_cache_layer_bias(lazy_cache, layer, i),
            )

    def test_extraction_parity(self):
        from aeon_core.splits import extract_cross_taxa_attentions_and_embeddings
        model = _model()
        dist, mds = _tree_inputs(seed=2)
        c, a = _site_batch(num_codons=20, seed=7)
        static_cache = model.precompute_tree_cache(dist, mds)
        lazy_cache = model.precompute_tree_cache(dist, mds, max_memory_mb=0.0)
        attn_s, repr_s = extract_cross_taxa_attentions_and_embeddings(
            model, c, a, static_cache, device="cpu")
        attn_l, repr_l = extract_cross_taxa_attentions_and_embeddings(
            model, c, a, lazy_cache, device="cpu")
        np.testing.assert_allclose(attn_s, attn_l, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(repr_s, repr_l, rtol=1e-5, atol=1e-6)
