#!/usr/bin/env python3
"""Model-based cross-alignment coselection (prototype).

Uses the hyphaeon PhyloAxialTransformer instead of raw association stats.
The trick (per hpi_integration_plan.md): put *individuals* on the taxon axis
and feed the model a TN93 distance matrix over them — no true phylogeny
needed. The transformer's tree-RoPE positional encoding then makes the
per-site attributions relatedness-aware for free.

Pipeline
--------
1. Concatenate each individual's host CDS + parasite CDS into one sequence;
   leaves = individuals. (one representative seq per individual per side)
2. load_alignment_and_tree(use_tn93=True) -> codon/aa tensors + TN93 d_mat
   + MDS z_coords -> model.precompute_tree_cache.
3. compute_transformer_attributions -> leaf_attributions [L_sites, N_inds].
4. compute_branch_coselection_network -> site x site cosine network; we keep
   only the host x parasite cross-block (sites < L_host vs >= L_host).

Requires aeon-core + hyphaeon + torch and a pretrained checkpoint (HF
variant or --weights). Input MSAs must be in-frame codon alignments.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from Bio import SeqIO


def read_side(fasta):
    """seq_id -> (sample_id, seq). Header {sample}|{gene}[|tag]; skips ref."""
    d = {}
    for r in SeqIO.parse(fasta, "fasta"):
        parts = r.id.split("|")
        if len(parts) >= 3 and parts[-1] == "ref":
            continue
        d[r.id] = (parts[0], str(r.seq).upper())
    return d


def build_concat(host_fa, par_fa, out_fa):
    """One concatenated sequence per shared individual; returns (n_inds, L_host_codons)."""
    host = read_side(host_fa)
    par = read_side(par_fa)
    # one representative seq per individual per side (first seen)
    h_rep, p_rep = {}, {}
    for sid, (samp, s) in host.items():
        h_rep.setdefault(samp, s)
    for sid, (samp, s) in par.items():
        p_rep.setdefault(samp, s)
    shared = sorted(set(h_rep) & set(p_rep))
    if not shared:
        sys.exit("no shared sample_ids between host and parasite MSAs")
    L_host_nt = len(h_rep[shared[0]])
    with open(out_fa, "w") as f:
        for s in shared:
            f.write(f">{s}\n{h_rep[s]}{p_rep[s]}\n")
    return shared, L_host_nt // 3


def main():
    ap = argparse.ArgumentParser(description="Model-based cross-alignment coselection")
    ap.add_argument("--host-aln", required=True, help="Host codon MSA (FASTA)")
    ap.add_argument("--par-aln", required=True, help="Parasite codon MSA (FASTA)")
    ap.add_argument("--weights", default=None, help="Model checkpoint path (else HF variant)")
    ap.add_argument("--variant", default=None, help="HF model variant name")
    ap.add_argument("--min-sim", type=float, default=0.35, help="Min attribution cosine")
    ap.add_argument("--dist-mode", default="balanced",
                    choices=["concat", "balanced", "par", "host"],
                    help="How to build the leaf distance matrix. 'concat' = TN93 on the "
                         "whole concatenated seq (faster-evolving side dominates). "
                         "'balanced' = mean of each side's normalized p-distance (equal "
                         "weight regardless of rate). 'par'/'host' = that side only.")
    ap.add_argument("--permute", type=int, default=0,
                    help="Permutation null: shuffle the parasite-site leaf axis P times "
                         "(breaks host x par pairing, preserves par x par) -> empirical p")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for permutations")
    ap.add_argument("--out", default=None, help="Output TSV of host x parasite pairs")
    args = ap.parse_args()

    try:
        import torch
        from aeon_core.inference import load_model, get_device
        from aeon_core.dataset import load_alignment_and_tree
        from hyphaeon.epistasis import (
            compute_transformer_attributions,
            compute_branch_coselection_network,
        )
    except ImportError as e:
        sys.exit(f"model coselection needs aeon-core + hyphaeon + torch: {e}")

    with tempfile.TemporaryDirectory() as td:
        concat = Path(td) / "concat.fasta"
        inds, L_host = build_concat(args.host_aln, args.par_aln, concat)
        print(f"[*] {len(inds)} individuals | host codons={L_host}")

        device = get_device()
        c, a, d_mat, z, inv_mask, taxa, L = load_alignment_and_tree(
            str(concat), use_tn93=True)
        # Rate-aware distance: the naive concat TN93 lets the faster-evolving
        # side dominate. For non-concat modes, recompute per-side p-distances
        # on the concat alignment and combine so neither side's rate swamps
        # the relatedness signal used for positional encoding.
        if args.dist_mode != "concat":
            d_mat, z = _side_distance(concat, taxa, L_host * 3, args.dist_mode)
        model = load_model(weights=args.weights, variant=args.variant, device=device)
        tree_cache = model.precompute_tree_cache(d_mat.to(device), z.to(device))
        print(f"[*] model loaded | codon sites={L} | running attributions...")

        attrs, lrts, pvals, cons = compute_transformer_attributions(
            model, c, a, tree_cache, taxa, device)
        # Mean-center each site's attribution across leaves: the raw vectors
        # share a global component (mean cosine ~0.39) that would otherwise
        # make most pairs look correlated. Centering isolates site-specific
        # coselection from the conservation/selection floor.
        attrs = np.asarray(attrs) - np.asarray(attrs).mean(axis=1, keepdims=True)
        edges, G = compute_branch_coselection_network(
            attrs, lrts, taxa, cons, min_sim=args.min_sim)

        # Permutation null: shuffle the leaf (individual) axis of the
        # parasite-site attribution rows only. This breaks host x parasite
        # pairing while preserving each side's marginal structure and the
        # par x par correlation — the correct null for cross-block coselection.
        obs_cross = _cross_edges(edges, L_host)
        if args.permute > 0:
            rng = np.random.default_rng(args.seed)
            null_counts, null_hub = [], []
            for _ in range(args.permute):
                perm = rng.permutation(attrs.shape[1])
                a_perm = attrs.copy()
                a_perm[L_host:] = a_perm[L_host:][:, perm]
                e_perm, _ = compute_branch_coselection_network(
                    a_perm, lrts, taxa, cons, min_sim=args.min_sim)
                ce = _cross_edges(e_perm, L_host)
                null_counts.append(len(ce))
                null_hub.append(_max_hub(ce))
            null_counts = np.array(null_counts)
            null_hub = np.array(null_hub)
            emp_p = (1 + int((null_counts >= len(obs_cross)).sum())) / (1 + args.permute)
            obs_hub = _max_hub(obs_cross)
            hub_p = (1 + int((null_hub >= obs_hub).sum())) / (1 + args.permute)
            print(f"[*] Permutation null ({args.permute}x): "
                  f"cross-edges null mean={null_counts.mean():.1f} max={null_counts.max()} "
                  f"| observed={len(obs_cross)} -> p={emp_p:.3g}")
            print(f"[*] Hub convergence (max edges on one par site): "
                  f"null mean={null_hub.mean():.1f} max={null_hub.max()} "
                  f"| observed={obs_hub} -> p={hub_p:.3g}")

    # keep only host x parasite cross-block edges (sites are 1-based;
    # host codons = 1..L_host, parasite codons = L_host+1..L_total)
    cross = obs_cross
    cross.sort(key=lambda e: e.get("fdr_q", 1.0))
    print(f"[✓] {len(edges)} network edges | {len(cross)} host x parasite cross edges")
    for e in cross[:25]:
        print(f"   host:{e['host_site']:<4} x par:{e['par_site']:<4} "
              f"sim={e['similarity']:.3f} q={e['fdr_q']:.3g} cesi={e.get('cesi', 0):.2f}")
    if args.out:
        import csv
        keys = sorted({k for e in cross for k in e})
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=keys)
            w.writeheader(); w.writerows(cross)
        print(f"    -> {args.out}")


def _pdist(seqs):
    """Pairwise p-distance (fraction of valid sites differing) for a list
    of equal-length sequences."""
    charmap = {}
    M = np.array([[charmap.setdefault(c, len(charmap)) if c not in "-NXnx?" else -1
                   for c in s] for s in seqs])
    valid = M >= 0
    n = len(M)
    D = np.zeros((n, n), np.float32)
    for i in range(n):
        both = valid[i] & valid[i + 1:]
        comp = both.sum(axis=1)
        diff = ((M[i] != M[i + 1:]) & both).sum(axis=1)
        d = np.where(comp > 0, diff / np.maximum(comp, 1), 0.0)
        D[i, i + 1:] = d
        D[i + 1:, i] = d
    return D


def _side_distance(concat_fa, taxa, host_nt, mode):
    """Per-side distance matrix + MDS coords for the concat alignment.

    Splits each taxon's concatenated sequence at host_nt into host/parasite
    portions, computes a p-distance on each, and combines per mode so the
    faster-evolving side does not dominate the relatedness estimate.
    """
    import torch
    from aeon_core.dataset import compute_mds_coordinates
    seqs = {r.id: str(r.seq).upper() for r in SeqIO.parse(concat_fa, "fasta")}
    h = _pdist([seqs[t][:host_nt] for t in taxa])
    p = _pdist([seqs[t][host_nt:] for t in taxa])
    if mode == "par":
        D = p
    elif mode == "host":
        D = h
    else:  # balanced: equal weight to each side's relatedness
        D = (h / max(h.max(), 1e-9) + p / max(p.max(), 1e-9)) / 2.0
    z = compute_mds_coordinates(D, n_components=4)
    return torch.tensor(D), torch.tensor(z, dtype=torch.float32)


def _max_hub(cross):
    """Max number of cross edges converging on a single parasite site —
    the 'hub' convergence signal (a real interaction locus draws many host
    sites to one parasite site)."""
    from collections import Counter
    if not cross:
        return 0
    return max(Counter(e["par_site"] for e in cross).values())


def _cross_edges(edges, L_host):
    """Return host x parasite cross-block edges (1-based sites; host codons
    = 1..L_host, parasite codons = L_host+1..L_total)."""
    out = []
    for e in edges:
        lo, hi = min(e["site_u"], e["site_v"]), max(e["site_u"], e["site_v"])
        if lo <= L_host < hi:
            e = dict(e)
            e["host_site"], e["par_site"] = lo, hi - L_host
            out.append(e)
    return out


if __name__ == "__main__":
    main()
