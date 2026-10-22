"""
hyphaeon/inference.py
---------------------
Site-level LRT prediction (HyphAeon-specific).

Generic inference helpers (device selection, model loading, alignment
preparation, adaptive batch sizing) now live in aeon_core.inference.
"""

import numpy as np
import torch

from aeon_core._progress import ChunkProgress


def predict_site_lrts(model, c, a, d, z, inv, tree_cache=None,
                      batch_size=64, device=None, progress=True, desc="Predict",
                      max_memory_mb=None):
    """Run model.forward_cached on variable sites only; return LRT array [L].

    Invariable sites get LRT=0. Uses tree_cache if provided, otherwise
    precomputes it.
    """
    if device is None:
        device = next(model.parameters()).device

    L = c.shape[0]
    lrts = np.zeros(L, dtype=np.float32)
    var_idx = np.where(~inv)[0]
    num_variable = len(var_idx)
    if num_variable == 0:
        return lrts

    if tree_cache is None:
        tree_cache = model.precompute_tree_cache(d.to(device), z.to(device), max_memory_mb=max_memory_mb)

    pb = ChunkProgress(num_variable, desc, 'codon', enabled=progress and num_variable > 0)
    with torch.no_grad():
        for s in range(0, num_variable, batch_size):
            end_idx = min(s + batch_size, num_variable)
            idx = var_idx[s:end_idx]
            y, _ = model.forward_cached(c[idx].to(device), a[idx].to(device), tree_cache)
            lrts[idx] = torch.clamp(y.squeeze(-1), min=0.0).cpu().numpy().flatten()
            pb.update(end_idx)
    pb.finish()

    return lrts
