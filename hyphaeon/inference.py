"""
hyphaeon/inference.py
---------------------
Shared inference helpers: device selection, model loading, alignment
preparation, and batched site-level LRT prediction.

These are used by the CLI subcommands (cmd_meme, cmd_busted, etc.),
hyphaeon/phenotype.py, hyphaeon/epistasis.py, hyphaeon/disease.py,
hyphaeon/filter.py, and model_eval/_harness.py.
"""

import numpy as np
import torch

from .model import PhyloAxialTransformer
from .weights import resolve_weights_path, load_arch_config, load_weights
from .dataset import load_alignment_and_tree


def get_device(cpu: bool = False) -> torch.device:
    """Select the best available hardware device.

    CUDA > MPS > CPU, unless cpu=True forces CPU.
    """
    if cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(weights=None, variant=None, device=None, strict=False):
    """Load a PhyloAxialTransformer from weights path or HuggingFace variant.

    Returns an eval-mode model on the specified device.
    """
    if device is None:
        device = get_device()

    weights_path = resolve_weights_path(weights=weights, variant=variant)
    config = load_arch_config(weights=weights, variant=variant)
    model = PhyloAxialTransformer(
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        window_size=config['window_size'],
    ).to(device)
    state_dict = load_weights(weights=weights_path, variant=variant, map_location=device)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model


def prepare_alignment(alignment_path, tree_path=None, model=None, device=None,
                      max_species=None, prune_duplicates=True):
    """Load an alignment + tree and precompute the tree attention cache.

    Returns (c, a, d, z, inv, taxa, L, tree_cache).
    If model is None, tree_cache will be None.
    """
    c, a, d, z, inv, taxa, L = load_alignment_and_tree(
        alignment_path, tree_path, max_species=max_species,
        prune_duplicates=prune_duplicates
    )
    tree_cache = None
    if model is not None:
        tree_cache = model.precompute_tree_cache(d.to(device), z.to(device))
    return c, a, d, z, inv, taxa, L, tree_cache


def predict_site_lrts(model, c, a, d, z, inv, tree_cache=None,
                      batch_size=64, device=None):
    """Run model.forward_cached on variable sites only; return LRT array [L].

    Invariable sites get LRT=0. Uses tree_cache if provided, otherwise
    precomputes it.
    """
    if device is None:
        device = next(model.parameters()).device

    L = c.shape[0]
    lrts = np.zeros(L, dtype=np.float32)
    var_idx = np.where(~inv)[0]
    if len(var_idx) == 0:
        return lrts

    if tree_cache is None:
        tree_cache = model.precompute_tree_cache(d.to(device), z.to(device))

    with torch.no_grad():
        for s in range(0, len(var_idx), batch_size):
            idx = var_idx[s:s + batch_size]
            y, _ = model.forward_cached(c[idx].to(device), a[idx].to(device), tree_cache)
            lrts[idx] = torch.clamp(y.squeeze(-1), min=0.0).cpu().numpy().flatten()

    return lrts
