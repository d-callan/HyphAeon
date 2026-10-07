"""Smoke test: verify all hyphaeon submodules import cleanly."""
import hyphaeon
from hyphaeon.phenotype import run_phenotype_association
from hyphaeon.epistasis import run_epistasis_analysis
from hyphaeon.disease import predict_disease_pathogenicity
from hyphaeon.filter import run_alignment_filter
from hyphaeon.evaluation import evaluate_directories
from hyphaeon.attribution import attribute_selection
from hyphaeon.training_data import build_training_directory
from hyphaeon.inference import predict_site_lrts
from hyphaeon.splits import spectral_bisection, run_spectral_splits
from hyphaeon.temporal import parse_temporal_metadata, infer_root_sequence


def test_version():
    assert hyphaeon.__version__ == "0.3.0"


def test_all_imports():
    assert run_phenotype_association is not None
    assert run_epistasis_analysis is not None
    assert predict_disease_pathogenicity is not None
    assert run_alignment_filter is not None
    assert evaluate_directories is not None
    assert attribute_selection is not None
    assert build_training_directory is not None
    assert predict_site_lrts is not None
    assert spectral_bisection is not None
    assert run_spectral_splits is not None
    assert parse_temporal_metadata is not None
    assert infer_root_sequence is not None
