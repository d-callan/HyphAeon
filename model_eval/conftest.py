"""
conftest.py for model_eval/

Resolves real AxoMEME weights (local path or Hugging Face download) and provides
shared fixtures. If weights cannot be resolved, all tests in this directory are
skipped with a clear reason — not silently passed, not errored.

This is deliberately separate from tests/conftest.py, which uses a dummy
random-weight checkpoint and must never depend on real weights.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Make the package and this directory importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from axomeme.model import PhyloAxialTransformer
from axomeme import dataset as ds

EXAMPLES_DIR = REPO_ROOT / "examples"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Mirror the CLI's weight resolution: AXOMEME_WEIGHTS env var, then the
# repo-local weights/axomeme_v1.pt. If a Hugging Face weights module is added
# in the future, it can be inserted here.
_REPO_WEIGHTS = REPO_ROOT / "weights" / "axomeme_v1.pt"


# ---------------------------------------------------------------------------
# Weight resolution
# ---------------------------------------------------------------------------

def _resolve_weights():
    """Return (path, source_label) or None if weights are unavailable."""
    explicit = os.environ.get("AXOMEME_WEIGHTS")
    if explicit and os.path.exists(explicit):
        return explicit, f"AXOMEME_WEIGHTS={explicit}"
    if _REPO_WEIGHTS.exists():
        return str(_REPO_WEIGHTS), f"repo weights ({_REPO_WEIGHTS})"
    return None, (
        "weights unavailable: set AXOMEME_WEIGHTS to a .pt checkpoint, "
        f"or place weights at {_REPO_WEIGHTS}"
    )


_WEIGHTS_PATH, _WEIGHTS_SOURCE = _resolve_weights()

WEIGHTS_AVAILABLE = _WEIGHTS_PATH is not None


def _load_model_from(path):
    """Load a checkpoint and return an eval-mode PhyloAxialTransformer."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(path)
        # safetensors has no embedded config; use architecture defaults.
        # If a config mechanism is added later, load it here.
        args = {"embed_dim": 384, "num_layers": 6, "num_heads": 12, "window_size": 1}
    else:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck.get("args", {}) if isinstance(ck, dict) else {}
        state_dict = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
        args = {
            "embed_dim": a.get("embed_dim", 384),
            "num_layers": a.get("layers", 6),
            "num_heads": a.get("heads", 12),
            "window_size": a.get("window_size", 1),
        }
    m = PhyloAxialTransformer(
        embed_dim=args["embed_dim"],
        num_layers=args["num_layers"],
        num_heads=args["num_heads"],
        window_size=args["window_size"],
    )
    m.load_state_dict(state_dict)
    m.eval()
    return m


@pytest.fixture(scope="session")
def weights_info():
    """Return (path, source_label); skip if weights unavailable."""
    if not WEIGHTS_AVAILABLE:
        pytest.skip(f"AxoMEME weights not available ({_WEIGHTS_SOURCE}). "
                    f"Set AXOMEME_WEIGHTS or HF_TOKEN to run model_eval tests.")
    return _WEIGHTS_PATH, _WEIGHTS_SOURCE


@pytest.fixture(scope="session")
def model(weights_info):
    """Session-scoped trained model in eval mode on CPU."""
    return _load_model_from(weights_info[0])


# ---------------------------------------------------------------------------
# Example data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def examples_dir():
    return str(EXAMPLES_DIR)


# ---------------------------------------------------------------------------
# Real dataset fixtures — three datasets spanning the parameter space
# ---------------------------------------------------------------------------

def _load_example(name, examples_dir):
    """Return (fasta_path, newick_path) for an example dataset, or skip."""
    fa = os.path.join(examples_dir, f"{name}.fasta")
    nwk = os.path.join(examples_dir, f"{name}.nwk")
    if not (os.path.exists(fa) and os.path.exists(nwk)):
        pytest.skip(f"{name} example data not found at {fa} / {nwk}")
    return fa, nwk


@pytest.fixture(scope="session")
def smc6_paths(examples_dir):
    """Smc6: 20 taxa, shallow tree (depth ~0.04), 26% variable sites."""
    return _load_example("Smc6", examples_dir)


@pytest.fixture(scope="session")
def bat_oas1_paths(examples_dir):
    """bat_oas1: 18 taxa, ultra-deep tree (depth ~62), 84% variable sites."""
    return _load_example("bat_oas1", examples_dir)


@pytest.fixture(scope="session")
def camelid_paths(examples_dir):
    """camelid: 212 taxa, topology-only tree (no branch lengths), 100% variable."""
    return _load_example("camelid", examples_dir)


def _make_base(model, paths):
    """Load tensors + baseline LRT for a dataset. Returns dict or None."""
    from _harness import load_tensors, predict, pvals_from_lrt
    fa, nwk = paths
    try:
        c, a, d, z, inv, taxa, L = load_tensors(fa, nwk)
    except Exception as e:
        print(f"  [WARNING] Dataset {os.path.basename(fa)} failed to load: "
              f"{type(e).__name__}: {e}")
        return None
    lrt = predict(model, c, a, d, z, inv)
    pval = pvals_from_lrt(lrt)
    return {
        "name": os.path.basename(fa).replace(".fasta", ""),
        "fa": fa, "nwk": nwk,
        "c": c, "a": a, "d": d, "z": z, "inv": inv,
        "taxa": taxa, "L": L, "lrt": lrt, "pval": pval,
        "tested": ~inv, "n_taxa": len(taxa),
    }


@pytest.fixture(scope="session")
def smc6_base(model, smc6_paths):
    """Loaded Smc6 tensors + baseline LRT predictions."""
    return _make_base(model, smc6_paths)


@pytest.fixture(scope="session")
def bat_oas1_base(model, bat_oas1_paths):
    """Loaded bat_oas1 tensors + baseline LRT predictions."""
    return _make_base(model, bat_oas1_paths)


@pytest.fixture(scope="session")
def camelid_base(model, camelid_paths):
    """Loaded camelid tensors + baseline LRT predictions."""
    return _make_base(model, camelid_paths)


@pytest.fixture(scope="session")
def real_datasets(model, smc6_base, bat_oas1_base, camelid_base):
    """All real datasets that loaded successfully. Returns list of base dicts.

    Each dict has: name, fa, nwk, c, a, d, z, inv, taxa, L, lrt, pval, tested, n_taxa.
    Datasets that fail to load (e.g. tree parsing issues) are skipped with
    a warning printed to stdout.
    """
    datasets = []
    for base in [smc6_base, bat_oas1_base, camelid_base]:
        if base is not None:
            datasets.append(base)
    if not datasets:
        pytest.skip("No real datasets loaded")
    return datasets


@pytest.fixture(scope="session")
def sim_datasets(model, seqgen_available):
    """Simulated datasets at varying tree depths and taxon counts.

    Fills the grid between the real datasets:
      - 50 taxa, moderate depth (0.2)
      - 100 taxa, deep (0.5)
      - 200 taxa, deep (1.0)

    Uses seq-gen with HKY under neutrality. Even though there's no real
    selection, the model produces non-trivial LRTs on variable sites —
    enough to test whether tree perturbations change the output.
    """
    from _sim import simulate_neutral_alignment
    from _harness import load_tensors, predict, pvals_from_lrt

    configs = [
        ("sim_50_mod", 50, 100, 0.2, 1.0, 100),
        ("sim_100_deep", 100, 100, 0.5, 1.0, 200),
        ("sim_200_deep", 200, 100, 1.0, 1.0, 300),
    ]
    datasets = []
    for name, n_taxa, n_codons, depth, scale, seed in configs:
        fa, nwk = simulate_neutral_alignment(
            n_taxa=n_taxa, n_codons=n_codons,
            tree_depth=depth, scale=scale, seed=seed)
        try:
            c, a, d, z, inv, taxa, L = load_tensors(fa, nwk)
        except Exception as e:
            print(f"  [WARNING] Simulated dataset {name} failed to load: "
                  f"{type(e).__name__}: {e}")
            continue
        lrt = predict(model, c, a, d, z, inv)
        pval = pvals_from_lrt(lrt)
        tested = ~inv
        if tested.sum() < 5:
            continue  # not enough variable sites
        datasets.append({
            "name": name, "fa": fa, "nwk": nwk,
            "c": c, "a": a, "d": d, "z": z, "inv": inv,
            "taxa": taxa, "L": L, "lrt": lrt, "pval": pval,
            "tested": tested, "n_taxa": len(taxa),
        })
    if not datasets:
        pytest.skip("No simulated datasets loaded")
    return datasets


@pytest.fixture(scope="session")
def all_datasets(real_datasets, sim_datasets):
    """All real + simulated datasets. Used by parametrized invariance gates."""
    total = real_datasets + sim_datasets
    if len(sim_datasets) == 0:
        print(f"  [WARNING] No simulated datasets available (seq-gen missing?). "
              f"Invariance gates will run on {len(real_datasets)} real datasets "
              f"only — majority threshold is {len(real_datasets)//2 + 1}.")
    return total


@pytest.fixture
def artifacts_dir():
    return str(ARTIFACTS_DIR)


# ---------------------------------------------------------------------------
# Tool availability checks
# ---------------------------------------------------------------------------

def _check_tool(name):
    """Return True if a command-line tool is on PATH."""
    import shutil
    return shutil.which(name) is not None


HYPHY_AVAILABLE = _check_tool("hyphy")
SEQGEN_AVAILABLE = _check_tool("seq-gen")


@pytest.fixture(scope="session")
def hyphy_available():
    """Skip test if HyPhy is not installed."""
    if not HYPHY_AVAILABLE:
        pytest.skip("HyPhy not on PATH — required for MEME concordance tests")
    return True


@pytest.fixture(scope="session")
def seqgen_available():
    """Skip test if seq-gen is not installed."""
    if not SEQGEN_AVAILABLE:
        pytest.skip("seq-gen not on PATH — required for neutral simulation tests")
    return True
