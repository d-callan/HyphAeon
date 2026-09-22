#!/usr/bin/env python3
"""Diffuse coselection: per-individual selection-burden correlation.

Why a separate method
---------------------
`model_coselect_paired.py --method delta_lrt` detects CONCENTRATED
selection-correlation: it needs specific lineages whose counterfactual
reversion moves a site's LRT, so it can ask "do the SAME lineages drive
selection at host site X and par site Y?". When selection is DIFFUSE —
the site LRT is produced by aggregate nonsynonymous diversity spread
across many independent lineages, no single allele/lineage responsible —
there is no per-lineage driver profile to correlate and delta_lrt returns
empty (the CR1 x PfRH4 result).

But diffuse selection can still co-vary at the INDIVIDUAL level: an
individual whose host gene carries an elevated load of selected/divergent
alleles may also carry an elevated load in the parasite gene (e.g. host
immune genotype <-> parasite escape genotype). This script computes a
per-individual selection-burden score for each gene and tests whether the
same individuals are burdened in both — a diffuse, individual-level
coselection signal that needs no concentrated site drivers.

Method
------
For each gene independently:
  1. dedup identical sequences -> unique haplotypes
  2. load_alignment_and_tree -> codon/aa tensors, TN93 distance, 4D MDS
     relatedness coords (z), per-site transformer attributions + LRTs
  3. per-individual burden[i] = sum over selected sites of the
     LRT-weighted attribution mass carried by i's haplotype
  4. regress burden on the MDS ancestry coords -> residuals remove the
     component of divergence explained by shared ancestry/relatedness
Then correlate host-residual vs par-residual across individuals and
compare to a permutation null (shuffle the par individual axis).
"""
import argparse, sys, tempfile
from pathlib import Path
import numpy as np
from Bio import SeqIO


def read_one_per_ind(path):
    """{sample}|{gene}|{tag} -> {sample: seq} (first record per sample)."""
    d = {}
    for r in SeqIO.parse(path, "fasta"):
        sid = r.id.split("|")[0]
        if sid not in d:
            d[sid] = str(r.seq).upper()
    return d


def side_burden(seqs, inds, model, device, tmpdir, tag, min_lrt):
    """Per-individual selection burden + ancestry coords for one MSA.

    Returns (burden [n_inds], resid [n_inds], n_sel_sites). burden[i] is the
    LRT-weighted attribution mass of i's haplotype summed over selected
    sites; resid is burden after regressing out the 4D MDS ancestry coords.
    """
    from aeon_core.dataset import load_alignment_and_tree
    from hyphaeon.epistasis import compute_transformer_attributions

    uniq_seqs, seen, ind2uniq = [], {}, []
    for ind in inds:
        s = seqs[ind]
        if s not in seen:
            seen[s] = len(uniq_seqs)
            uniq_seqs.append(s)
        ind2uniq.append(seen[s])
    ind2uniq = np.asarray(ind2uniq)

    fa = Path(tmpdir) / f"{tag}.fasta"
    with open(fa, "w") as f:
        for i, s in enumerate(uniq_seqs):
            f.write(f">u{i}\n{s}\n")

    c, a, d_mat, z, inv_mask, taxa, L = load_alignment_and_tree(str(fa), use_tn93=True)
    tree_cache = model.precompute_tree_cache(d_mat.to(device), z.to(device))
    attrs, lrts, _, _ = compute_transformer_attributions(
        model, c, a, tree_cache, taxa, device)
    attrs = np.asarray(attrs)          # [L, n_leaves] in `taxa` order
    lrts = np.asarray(lrts)            # [L]

    # map taxa order -> my unique order (taxa names are 'u{i}')
    taxa_idx = np.asarray([int(t.lstrip("u")) for t in taxa])
    attrs_uniq = np.zeros((attrs.shape[0], len(uniq_seqs)), dtype=attrs.dtype)
    for col, uidx in enumerate(taxa_idx):
        attrs_uniq[:, uidx] = attrs[:, col]

    sel = lrts >= min_lrt
    # LRT-weighted attribution mass per unique haplotype, summed over sel sites
    w = np.clip(lrts[sel], 0, None)[:, None]            # [n_sel, 1]
    burden_uniq = (attrs_uniq[sel] * w).sum(axis=0)     # [n_uniq]
    burden = burden_uniq[ind2uniq]                       # [n_inds]

    # Ancestry must be estimated from NON-selected sites only. The MDS coords
    # from the full alignment bake the selected-site divergence into "ancestry",
    # so regressing on them removes the very signal under test (a divergent
    # taxon is both high-burden AND far in MDS space). Rebuilding relatedness
    # from the neutral background isolates shared descent, letting genuine
    # non-ancestry (e.g. convergent / scattered) coupling survive.
    keep = np.where(~sel)[0]
    fa_neu = Path(tmpdir) / f"{tag}_neutral.fasta"
    with open(fa_neu, "w") as f:
        for i, s in enumerate(uniq_seqs):
            codons = [s[3*j:3*j+3] for j in keep]
            f.write(f">u{i}\n{''.join(codons)}\n")
    _, _, _, z_neu, _, taxa_neu, _ = load_alignment_and_tree(
        str(fa_neu), use_tn93=True)
    neu_idx = np.asarray([int(t.lstrip("u")) for t in taxa_neu])
    z_uniq = np.zeros((len(uniq_seqs), z_neu.shape[-1]))
    z_uniq[neu_idx] = z_neu[0].cpu().numpy()
    Z = z_uniq[ind2uniq]                                 # [n_inds, k]
    # regress burden on ancestry (with intercept) -> residuals
    X = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(X, burden, rcond=None)
    resid = burden - X @ beta
    return burden, resid, int(sel.sum())


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float((x @ y) / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host-aln", required=True)
    ap.add_argument("--par-aln", required=True)
    ap.add_argument("--min-lrt", type=float, default=1.0,
                    help="Min site LRT to count toward the burden")
    ap.add_argument("--permute", type=int, default=1000,
                    help="Permutations of the par axis for the null")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from aeon_core.inference import load_model, get_device
    device = get_device()
    model = load_model(device=device)

    host = read_one_per_ind(args.host_aln)
    par = read_one_per_ind(args.par_aln)
    inds = sorted(set(host) & set(par))
    print(f"[*] {len(inds)} shared individuals | host={len(host)} par={len(par)}")

    with tempfile.TemporaryDirectory() as td:
        bh, rh, nsh = side_burden(host, inds, model, device, td, "host", args.min_lrt)
        bp, rp, nsp = side_burden(par, inds, model, device, td, "par", args.min_lrt)
    print(f"[*] selected sites (lrt>={args.min_lrt}): host={nsh} par={nsp}")

    # raw burden correlation and ancestry-corrected (residual) correlation
    r_raw = pearson(bh, bp)
    r_resid = pearson(rh, rp)

    rng = np.random.default_rng(args.seed)
    null_raw = np.empty(args.permute)
    null_resid = np.empty(args.permute)
    for p in range(args.permute):
        perm = rng.permutation(len(inds))
        null_raw[p] = pearson(bh, bp[perm])
        null_resid[p] = pearson(rh, rp[perm])
    p_raw = (np.sum(null_raw >= r_raw) + 1) / (args.permute + 1)
    p_resid = (np.sum(null_resid >= r_resid) + 1) / (args.permute + 1)

    print(f"[*] RAW burden corr        : r={r_raw:+.4f}  null mean={null_raw.mean():+.4f}  p={p_raw:.4f}")
    print(f"[*] ANCESTRY-CORRECTED corr: r={r_resid:+.4f}  null mean={null_resid.mean():+.4f}  p={p_resid:.4f}")
    print("[i] resid = burden after regressing out per-gene MDS ancestry; "
          "a significant resid corr = same individuals carry excess selected "
          "divergence in both genes beyond what ancestry predicts.")


if __name__ == "__main__":
    main()
