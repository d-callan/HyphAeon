"""
HyphAeon: Ultra-Fast Neural Inference of Episodic Selection,
Phenotype-Genotype Association Mapping, and Epistatic Sector Mining.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from aeon_core.model import PhyloAxialTransformer
from aeon_core.dataset import load_alignment_and_tree, parse_alignment_sequences, compute_tn93_distance_matrix, parse_beast_xml
from .phenotype import run_phenotype_association, resolve_phenotype_vector, PRESETS
from .epistasis import (
    run_epistasis_analysis,
    run_digital_dms_analysis,
    run_dms_analysis,
    compute_phylogenetic_branch_attributions,
    compute_branch_coselection_network,
    compute_selection_dms_essm,
    extract_epistatic_sectors
)
from .disease import predict_disease_pathogenicity

try:
    __version__ = _pkg_version("hyphaeon")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+local"
__all__ = [
    "PhyloAxialTransformer",
    "load_alignment_and_tree",
    "compute_tn93_distance_matrix",
    "parse_alignment_sequences",
    "parse_beast_xml",
    "run_phenotype_association",
    "resolve_phenotype_vector",
    "run_epistasis_analysis",
    "run_digital_dms_analysis",
    "run_dms_analysis",
    "compute_phylogenetic_branch_attributions",
    "compute_branch_coselection_network",
    "compute_selection_dms_essm",
    "extract_epistatic_sectors",
    "predict_disease_pathogenicity",
    "PRESETS",
]
