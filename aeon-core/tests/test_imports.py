"""Smoke test: verify all aeon_core submodules import cleanly."""
import aeon_core
from aeon_core.model import PhyloAxialTransformer, BustedMultiTaskHead
from aeon_core.weights import resolve_weights_path, load_model_config
from aeon_core.inference import get_device, compute_adaptive_safe_batch_size
from aeon_core.splits import compute_fused_affinity_matrix
from aeon_core.dataset import parse_alignment_sequences, load_alignment_and_tree
from aeon_core.temporal import parse_date_to_decimal, extract_date_from_string
from aeon_core.io import write_json, write_csv
from aeon_core.stats import benjamini_hochberg, cauchy_combination_p
from aeon_core._progress import ChunkProgress


def test_version():
    assert aeon_core.__version__ == "0.2.0"


def test_all_imports():
    assert PhyloAxialTransformer is not None
    assert BustedMultiTaskHead is not None
    assert resolve_weights_path is not None
    assert get_device is not None
    assert compute_fused_affinity_matrix is not None
    assert parse_alignment_sequences is not None
    assert parse_date_to_decimal is not None
    assert write_json is not None
    assert benjamini_hochberg is not None
    assert ChunkProgress is not None
