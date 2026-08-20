"""
Input diagnostics: the model's preprocessing pipeline must handle common
input variations gracefully. These are not phylogeny-sensitivity gates — they
check that the data pipeline doesn't silently degrade into garbage on
realistic inputs.

Tests:
  - U instead of T (RNA alphabet): the pipeline normalizes U→T during
    parsing, so U-format input should produce the same tokens as T-format.
  - Frameshift by 1nt: codons are misaligned, producing nonsense tokens.
    The pipeline does not detect or warn about frame errors — this test
    documents that the model runs silently on misaligned input.
"""
import numpy as np
import pytest

from _harness import load_tensors, make_frameshift_alignment, make_uracil_alignment


UNKNOWN_CODON_TOKEN = 64
UNKNOWN_AA_TOKEN = 20


class TestAlphabetHandling:
    """The codon table uses T, not U. RNA-format alignments (U instead of T)
    must either be handled or rejected — not silently tokenized as unknown.

    The dataset pipeline normalizes U→T during parsing
    (`.replace('U', 'T')` in parse_alignment_sequences), so U-format input
    should produce the same tokens as T-format input. This test verifies that
    normalization works correctly.
    """

    def test_uracil_normalized_to_thymine(self, model, smc6_base, smc6_paths, tmp_path):
        """U-format alignments should produce the same codon tokens as T-format,
        not a flood of unknown tokens.
        """
        base = smc6_base
        fa, nwk = smc6_paths
        ura_fa = make_uracil_alignment(fa, base["taxa"], tmp_path)
        c, a, d, z, inv, taxa, L = load_tensors(ura_fa, nwk)
        codon_tokens = c[:, :, 0].numpy()
        unknown_frac = float((codon_tokens == UNKNOWN_CODON_TOKEN).mean())
        # The baseline (T-format) unknown rate should be near zero too.
        base_unknown = float((base["c"][:, :, 0].numpy() == UNKNOWN_CODON_TOKEN).mean())
        assert unknown_frac <= base_unknown + 0.01, (
            f"U-format unknown token rate ({unknown_frac:.1%}) is much higher "
            f"than T-format ({base_unknown:.1%}). U→T normalization may be broken."
        )


class TestFrameshiftHandling:
    """A 1-nucleotide frameshift misaligns all codons. The pipeline trims
    trailing nucleotides to a multiple of 3 but does not detect or warn about
    frame errors. This test documents that the model produces predictions on
    misaligned codons without complaint.
    """

    def test_frameshift_runs_silently(self, model, smc6_base, smc6_paths, tmp_path):
        base = smc6_base
        fa, nwk = smc6_paths
        fs_fa = make_frameshift_alignment(fa, base["taxa"], tmp_path, shift=1)
        c, a, d, z, inv, taxa, L = load_tensors(fs_fa, nwk)
        # The model runs and produces output — the issue is that it shouldn't,
        # at least not without a warning. Document that it does.
        from _harness import predict
        lrt = predict(model, c, a, d, z, inv)
        assert np.isfinite(lrt).all(), "Frameshift produced non-finite LRTs"
        # The LRT distribution will differ from baseline because the codons
        # are different. The point is the pipeline didn't reject the input.
