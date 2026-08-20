"""
Numerical edge cases: degenerate inputs that should not crash or produce
garbage. These test the model's robustness to unusual but realistic data.
"""
import numpy as np
import pytest

from _harness import load_tensors, predict


class TestDegenerateColumns:
    """Columns with all gaps, single polymorphism, or all-same codon should
    not crash the model or produce non-finite output.
    """

    def test_all_gap_column(self, model, smc6_base, smc6_paths, tmp_path):
        """Replace one site with all-gap codons (token 64). The model should
        handle this gracefully — the site will be marked invariable, but the
        forward pass should not crash.
        """
        base = smc6_base
        c = base["c"].clone()
        # Set site 5 to all unknown codons
        c[5, :, 0] = 64
        lrt = predict(model, c, base["a"], base["d"], base["z"], base["inv"])
        assert np.isfinite(lrt).all(), "All-gap column produced non-finite LRT"

    def test_single_polymorphism(self, model, smc6_base):
        """A site where only one taxon differs should produce a valid LRT.
        This is the minimal case for positive selection detection.
        """
        base = smc6_base
        c = base["c"].clone()
        a = base["a"].clone()
        # At site 10, make taxon 0 differ from all others.
        # Use a codon token that differs from the original to ensure
        # we actually create a polymorphism (not a no-op).
        orig_token = c[10, 0, 0].item()
        new_token = 1 if orig_token != 1 else 2  # TTC or TTT (both Phe)
        c[10, 0, 0] = new_token
        a[10, 0, 0] = 1  # Phe AA token
        # Verify the change actually created a polymorphism
        site_tokens = c[10, :, 0].numpy()
        assert len(set(site_tokens)) > 1, (
            f"Site 10 has no polymorphism after edit — all tokens are "
            f"{set(site_tokens)}. The test is a no-op.")
        lrt = predict(model, c, a, base["d"], base["z"], base["inv"])
        assert np.isfinite(lrt).all(), "Single polymorphism produced non-finite LRT"
        assert lrt[10] >= 0, "LRT should be non-negative"


class TestSmallAlignment:
    """The model should handle very small alignments without crashing.
    The minimum useful alignment is 2 taxa, but the pipeline may enforce
    a higher minimum.
    """

    def test_two_taxa(self, model, tmp_path):
        """2 taxa, 10 codons. This is degenerate but should not crash."""
        fa = tmp_path / "small.fasta"
        fa.write_text(">t1\nATGTTTCTTGGTATGTTTCTTGGTATGTTTCTTGGTATGTTTCTTGGTATGTTTCTTGGT\n"
                      ">t2\nATGTTTCTTGGTATGTTTCTCGGTATGTTTCTTGGTATGTTTCTTGGTATGTTTCTTGGT\n")
        nwk = tmp_path / "small.nwk"
        nwk.write_text("(t1:0.01,t2:0.01);\n")
        try:
            c, a, d, z, inv, taxa, L = load_tensors(str(fa), str(nwk))
            lrt = predict(model, c, a, d, z, inv)
            assert np.isfinite(lrt).all(), "2-taxon alignment produced non-finite LRT"
        except Exception as e:
            # The pipeline may reject very small alignments — that's OK,
            # but it should fail with a clear error, not a cryptic crash.
            assert "taxa" in str(e).lower() or "species" in str(e).lower(), (
                f"2-taxon alignment failed with unexpected error: {e}"
            )


class TestLargeTaxaCount:
    """The model should handle alignments with many taxa without crashing
    or degrading catastrophically. The training corpus includes up to 2,941
    taxa, but the model may have practical limits.
    """

    def test_100_taxa(self, model, seqgen_available, tmp_path):
        """100 taxa, 50 codons. Tests that the model scales to moderate N."""
        from _sim import simulate_neutral_alignment
        fa, nwk = simulate_neutral_alignment(
            n_taxa=100, n_codons=50, seed=77, scale=0.5)
        c, a, d, z, inv, taxa, L = load_tensors(fa, nwk)
        lrt = predict(model, c, a, d, z, inv)
        assert np.isfinite(lrt).all(), "100-taxon alignment produced non-finite LRT"
