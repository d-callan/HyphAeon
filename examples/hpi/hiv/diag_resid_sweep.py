"""Diagnostic: sweep ancestry-correction strength for the screen's resid.

The screen's primary signal is burden_r_resid = corr(resid_h, resid_p) where
resid = burden after regressing out neutral-site ancestry (MDS coords). The
correction is currently a FIXED 4-dim MDS + OLS. Two failure modes showed up
on the simulated panel:

  * under-correction: a clade gene's burden is ancestry; if 4 MDS dims don't
    capture the clade, resid keeps clade structure -> spurious resid coupling
    (h_act1 x p_act2 = 0.298).
  * over-correction: OLS on the MDS dims can absorb part of a genuinely
    scattered (non-ancestry) signal -> weak resid on a true positive
    (h_scatA x p_scatA = 0.062 vs h_scatB x p_scatB = 0.461).

This caches the expensive parts (per-gene transformer pass + neutral TN93
matrix) ONCE, then sweeps n_mds_dims x {ols, ridge} cheaply and reports the
resid correlation for the diagnostic pairs.

Run: python diag_resid_sweep.py PANEL_DIR
"""
import os
import sys
import glob
import tempfile

import numpy as np

from scipy.stats import rankdata

from symbaeon.pairing import load_paired_alignment
from symbaeon.model import _load_model, _representative_seqs, _side_base, _pearson
from aeon_core.dataset import (compute_tn93_distance_matrix,
                               compute_mds_coordinates)


def _spearman(x, y):
    return _pearson(rankdata(x), rankdata(y))

MIN_LRT = 1.0

# pairs whose resid we inspect: (host, par, want)  want=high|low
PAIRS = [
    ("h_scatA", "p_scatA", "HIGH"),   # true non-ancestry coupled
    ("h_scatB", "p_scatB", "HIGH"),   # true non-ancestry coupled
    ("h_cladeA", "p_cladeA", "low"),  # ancestry-driven -> resid should vanish
    ("h_act1", "p_act2", "low"),      # independent-active -> no coupling
    ("h_act2", "p_scatA", "low"),     # independent-active -> no coupling
]


def _burden(sd, min_lrt):
    sel = sd.lrts >= min_lrt
    w = np.clip(sd.lrts[sel], 0, None)[:, None]
    return (sd.attrs_uniq[sel] * w).sum(axis=0)[sd.ind2uniq]


def _neutral_distmat(sd, min_lrt):
    keep = np.where(sd.lrts < min_lrt)[0]
    seq_dict = {}
    for i, s in enumerate(sd.uniq_seqs):
        seq_dict[f"u{i}"] = "".join(s[3 * j:3 * j + 3] for j in keep)
    taxa = list(seq_dict)
    return compute_tn93_distance_matrix(seq_dict, taxa), taxa


def _resid_knn(burden, dm_ind, k):
    """Relatedness-kernel residual: predict each individual's burden from its
    k nearest neighbours in neutral-distance space (leave-one-out). Removes
    burden shared among RELATIVES (clade) while preserving isolated/group
    elevations that aren't ancestry (scattered) — unlike MDS-linear regression,
    which absorbs any group that is linearly separable in embedding space."""
    n = len(burden)
    resid = np.empty(n)
    for i in range(n):
        d = dm_ind[i].astype(float).copy()
        d[i] = np.inf
        nn = np.argsort(d)[:k]
        resid[i] = burden[i] - burden[nn].mean()
    return resid


def _resid(burden, Z, method, lam):
    X = np.column_stack([np.ones(len(Z)), Z])
    if method == "ridge":
        A = X.T @ X + lam * np.eye(X.shape[1])
        A[0, 0] -= lam                      # don't penalize intercept
        beta = np.linalg.solve(A, X.T @ burden)
    elif method == "loo":
        # leave-one-out CV prediction error: resid_i = burden_i - fit(others)_i.
        # In-sample OLS can absorb a NON-ancestry group's elevation when the
        # group's neutral coords give a chance linear fit (overcorrection ->
        # kills true scattered signal). Out-of-fold prediction only removes
        # what ancestry genuinely predicts, so scattered signal survives.
        resid = np.empty(len(burden))
        for i in range(len(burden)):
            m = np.ones(len(burden), bool); m[i] = False
            beta, *_ = np.linalg.lstsq(X[m], burden[m], rcond=None)
            resid[i] = burden[i] - X[i] @ beta
        return resid
    else:                                   # ols
        beta, *_ = np.linalg.lstsq(X, burden, rcond=None)
    return burden - X @ beta


def main():
    panel = sys.argv[1] if len(sys.argv) > 1 else "/tmp/panel"
    host_f = sorted(glob.glob(os.path.join(panel, "host", "*.fasta")))
    par_f = sorted(glob.glob(os.path.join(panel, "par", "*.fasta")))
    sides = {}
    for p in host_f + par_f:
        name = os.path.splitext(os.path.basename(p))[0]
        role = "host" if "/host/" in p else "parasite"
        sides[name] = load_paired_alignment(p, role, None)
    inds = sorted(set.intersection(*[set(s.sample_to_seqs)
                                     for s in sides.values()]))
    print(f"[*] {len(sides)} genes, {len(inds)} shared individuals")

    model, device = _load_model(None, None)
    cache = {}
    with tempfile.TemporaryDirectory() as td:
        for name, side in sides.items():
            seqs = _representative_seqs(side, inds)
            sd = _side_base(seqs, inds, model, device, td, name)
            dm, taxa = _neutral_distmat(sd, MIN_LRT)
            cache[name] = dict(burden=_burden(sd, MIN_LRT),
                               ind2uniq=sd.ind2uniq, dm=dm, taxa=taxa)
            print(f"[*] cached {name}")

    # ---- sweep: OLS-MDS baseline vs kNN relatedness-kernel resid ----
    print("\nresid correlation per pair (want HIGH for coupled, low for rest)")
    hdr = ("method  param " +
           "".join(f"{h[:4]}-{p[:4]:>11s}" for h, p, _ in PAIRS) +
           "   minTP-maxFP")
    print(hdr)

    def report(tag, resid):
        vals = [_pearson(resid[h], resid[p]) for h, p, _ in PAIRS]
        sep = min(vals[0], vals[1]) - max(vals[2], vals[3], vals[4])
        cells = "".join(f"{v:>11.3f}" for v in vals)
        print(f"{tag:<14}{cells}   {sep:+.3f}")

    # baseline: OLS on 4-dim MDS (current behaviour)
    Zmap = {n: compute_mds_coordinates(c["dm"], 4)[c["ind2uniq"]]
            for n, c in cache.items()}
    report("ols-mds4", {n: _resid(c["burden"], Zmap[n], "ols", 0.0)
                        for n, c in cache.items()})
    # kNN relatedness-kernel resid at several neighbourhood sizes
    for knn in (3, 5, 8, 12):
        resid = {}
        for n, c in cache.items():
            dm_ind = c["dm"][np.ix_(c["ind2uniq"], c["ind2uniq"])]
            resid[n] = _resid_knn(c["burden"], dm_ind, knn)
        report(f"knn-{knn}", resid)
        print()


if __name__ == "__main__":
    main()
