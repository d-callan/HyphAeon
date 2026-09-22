#!/usr/bin/env python3
"""Model-based cross-alignment coselection WITHOUT concatenation.

Unlike model_coselect.py (which concatenates host+parasite into one hybrid
sequence and computes a single TN93 distance — letting the faster-evolving
side dominate the relatedness estimate), this script runs the transformer
SEPARATELY on each MSA:

  host MSA -> host TN93 distance -> host tree_cache -> host attributions
  par  MSA -> par  TN93 distance -> par  tree_cache -> par  attributions

Each side gets its own distance matrix / positional encoding, so neither
side's substitution rate swamps the other. Coselection is then the cosine
between a host site's and a parasite site's attribution profiles across the
SHARED individual axis — the same cross-block signal as the concat version,
but with correct per-side relatedness.

The subtlety is the leaf axis: load_alignment_and_tree dedups identical
sequences, collapsing individuals differently per side. We dedup each side
ourselves (unique haplotypes + an individual->haplotype map), run the model
on the uniques, then expand attributions back to per-individual so both
sides share the same individual axis before the cross-cosine.
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


def side_attributions(seqs, inds, model, device, tmpdir, tag,
                      method="attn", attr_min_lrt=3.84):
    """Run the transformer on one side's MSA; return per-individual
    attributions [L, n_inds] aligned to `inds` order.

    Dedups identical sequences (the model only needs unique haplotypes),
    runs load_alignment_and_tree (its own TN93 distance -> tree_cache ->
    attributions), then expands back to per-individual via the dedup map.

    method:
      'attn'      -> alpha*delta leaf attributions (prediction structure)
      'delta_lrt' -> counterfactual Delta-LRT per (site,taxon): how much each
                     taxon's non-consensus codon drives the site selection LRT.
                     This is the selection-DRIVER profile — correlating it
                     across host x par tests selection-correlation (same
                     lineages drive selection at both sites), not just
                     prediction-structure similarity.
    """
    from aeon_core.dataset import load_alignment_and_tree
    from hyphaeon.epistasis import compute_transformer_attributions

    # dedup: unique sequences + ind -> unique-index map
    uniq_seqs, seen, ind2uniq = [], {}, []
    for ind in inds:
        s = seqs[ind]
        if s not in seen:
            seen[s] = len(uniq_seqs)
            uniq_seqs.append(s)
        ind2uniq.append(seen[s])

    fa = Path(tmpdir) / f"{tag}.fasta"
    with open(fa, "w") as f:
        for i, s in enumerate(uniq_seqs):
            f.write(f">u{i}\n{s}\n")

    c, a, d_mat, z, inv_mask, taxa, L = load_alignment_and_tree(str(fa), use_tn93=True)
    tree_cache = model.precompute_tree_cache(d_mat.to(device), z.to(device))
    attrs, lrts, _, _ = compute_transformer_attributions(
        model, c, a, tree_cache, taxa, device)
    attrs = np.asarray(attrs)  # [L, n_leaves] in `taxa` order

    # map taxa order -> my unique order (taxa names are 'u{i}')
    taxa_idx = [int(t.lstrip("u")) for t in taxa]

    if method == "delta_lrt":
        from hyphaeon.attribution import attribute_selection
        res = attribute_selection(
            model, c, a, d_mat, z, inv_mask, taxa,
            min_lrt=attr_min_lrt, base_lrts=np.asarray(lrts), cache=tree_cache)
        # [L, n_uniq] matrix of delta_lrt (0 for consensus taxa / unselected sites)
        attrs_uniq = np.zeros((L, len(uniq_seqs)), dtype=np.float32)
        for site, meta in res.items():
            for d_sp in meta["driving_species"]:
                attrs_uniq[site, taxa_idx[d_sp["taxon_index"]]] = d_sp["delta_lrt"]
    elif method == "delta_lrt_aa":
        # GROUP counterfactual: revert ALL carriers of each derived amino acid
        # jointly -> delta_lrt per allele-group, assigned to each carrier.
        # Single-taxon delta_lrt is underpowered when selection is diffuse across
        # many haplotypes (no single reversion moves the LRT). Reverting a whole
        # allele-class captures allele-driven selection; if even this yields no
        # drivers, the site LRT is driven by aggregate diversity, not a lineage.
        import torch as _t
        from aeon_core.dataset import CODON_TO_AA, AA_MAP, GENETIC_CODE
        INV = {v: k for k, v in GENETIC_CODE.items()}
        cn = c.cpu().numpy(); an = a.cpu().numpy()
        lrt_arr = np.asarray(lrts)
        attrs_uniq = np.zeros((L, len(uniq_seqs)), dtype=np.float32)
        for site in np.where(lrt_arr >= attr_min_lrt)[0]:
            site_lrt = float(lrt_arr[site])
            codons = cn[site, :, 0]; aas = an[site, :, 0]
            vals, counts = np.unique(codons, return_counts=True)
            cons_codon = int(vals[np.argmax(counts)])
            cons_aa = AA_MAP.get(CODON_TO_AA.get(INV.get(cons_codon, 'NNN'), '-'), 20)
            groups = {}
            for t in np.where(codons != cons_codon)[0]:
                groups.setdefault(int(aas[t]), []).append(int(t))
            for aa_tok, members in groups.items():
                c_mod = cn[site:site+1].copy(); a_mod = an[site:site+1].copy()
                c_mod[0, members, 0] = cons_codon
                a_mod[0, members, 0] = cons_aa
                with _t.no_grad():
                    y_mod, _ = model.forward_cached(
                        _t.from_numpy(c_mod).long().to(device),
                        _t.from_numpy(a_mod).long().to(device), tree_cache)
                mod_lrt = float(_t.clamp(y_mod.squeeze(-1), min=0.0).detach().cpu().numpy().ravel()[0])
                for t in members:
                    attrs_uniq[site, t] = site_lrt - mod_lrt
    else:
        attrs_uniq = np.zeros((attrs.shape[0], len(uniq_seqs)), dtype=attrs.dtype)
        for col, uidx in enumerate(taxa_idx):
            attrs_uniq[:, uidx] = attrs[:, col]

    # expand unique -> per-individual; lrts are per-site (no leaf axis)
    return attrs_uniq[:, np.asarray(ind2uniq)], np.asarray(lrts)


def cross_cosine(Ah, Ap):
    """Mean-center each site then cosine between every host-site and
    parasite-site attribution profile across individuals -> [L_h, L_p]."""
    Ah = Ah - Ah.mean(axis=1, keepdims=True)
    Ap = Ap - Ap.mean(axis=1, keepdims=True)
    Ah /= np.maximum(np.linalg.norm(Ah, axis=1, keepdims=True), 1e-12)
    Ap /= np.maximum(np.linalg.norm(Ap, axis=1, keepdims=True), 1e-12)
    return Ah @ Ap.T


def edges_from(C, min_sim, lrt_h=None, lrt_p=None, min_lrt=0.0, min_cesi=0.0):
    """(host_site, par_site, sim, cesi) for |cosine| >= min_sim AND both
    sites under selection (lrt >= min_lrt) AND cesi >= min_cesi.

    The min_lrt/CESI gate restricts to SELECTION-correlation (the model's
    actual target): both sites must be under selection, and the edge weight
    is scaled by both LRTs — mirroring compute_branch_coselection_network."""
    u, v = np.where(np.abs(C) >= min_sim)
    out = []
    for i, j in zip(u, v):
        sim = float(C[i, j])
        lu = float(lrt_h[i]) if lrt_h is not None else 1.0
        lv = float(lrt_p[j]) if lrt_p is not None else 1.0
        cesi = abs(sim) * np.sqrt(max(lu, 0.1) * max(lv, 0.1))
        if max(lu, lv) >= min_lrt and cesi >= min_cesi:
            out.append((int(i) + 1, int(j) + 1, sim, float(cesi)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host-aln", required=True, help="Host codon MSA (FASTA)")
    ap.add_argument("--par-aln", required=True, help="Parasite codon MSA (FASTA)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--method", choices=["attn", "delta_lrt", "delta_lrt_aa"],
                    default="attn",
                    help="attribution profile: attn=alpha*delta (prediction "
                         "structure) | delta_lrt=single-taxon counterfactual "
                         "driver | delta_lrt_aa=allele-group counterfactual "
                         "(revert all carriers of a derived AA jointly)")
    ap.add_argument("--attr-min-lrt", type=float, default=3.84,
                    help="Delta-LRT: min site LRT to attribute (focal-site gate)")
    ap.add_argument("--min-sim", type=float, default=0.5)
    ap.add_argument("--min-lrt", type=float, default=1.0,
                    help="Min selection LRT — at least one site must be under "
                         "selection (gates to selection-correlation)")
    ap.add_argument("--min-cesi", type=float, default=0.0,
                    help="Min CESI = |cos|*sqrt(lrt_u*lrt_v)")
    ap.add_argument("--permute", type=int, default=0,
                    help="Permutation null: shuffle parasite leaf axis P times")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        import torch
        from aeon_core.inference import load_model, get_device
    except Exception as e:
        sys.exit(f"needs aeon-core + hyphaeon + torch: {e}")

    host = read_one_per_ind(args.host_aln)
    par = read_one_per_ind(args.par_aln)
    inds = sorted(set(host) & set(par))
    if len(inds) < 4:
        sys.exit("too few shared individuals")
    print(f"[*] {len(inds)} shared individuals | host={len(host)} par={len(par)}")

    device = get_device()
    model = load_model(weights=args.weights, variant=args.variant, device=device)

    with tempfile.TemporaryDirectory() as td:
        Ah, lrt_h = side_attributions(host, inds, model, device, td, "host",
                                    args.method, args.attr_min_lrt)
        print(f"[*] host attributions {Ah.shape} | selected sites (lrt>={args.min_lrt}): "
              f"{(lrt_h >= args.min_lrt).sum()}")
        Ap, lrt_p = side_attributions(par, inds, model, device, td, "par",
                                    args.method, args.attr_min_lrt)
        print(f"[*] par attributions {Ap.shape} | selected sites (lrt>={args.min_lrt}): "
              f"{(lrt_p >= args.min_lrt).sum()}")

    C = cross_cosine(Ah, Ap)
    obs = edges_from(C, args.min_sim, lrt_h, lrt_p, args.min_lrt, args.min_cesi)

    if args.permute > 0:
        rng = np.random.default_rng(args.seed)
        null_n, null_hub = [], []
        for _ in range(args.permute):
            Cp = cross_cosine(Ah, Ap[:, rng.permutation(Ap.shape[1])])
            e = edges_from(Cp, args.min_sim, lrt_h, lrt_p, args.min_lrt, args.min_cesi)
            null_n.append(len(e))
            from collections import Counter
            null_hub.append(max(Counter(x[1] for x in e).values()) if e else 0)
        null_n, null_hub = np.array(null_n), np.array(null_hub)
        obs_hub = max(__import__("collections").Counter(x[1] for x in obs).values()) if obs else 0
        print(f"[*] perm null ({args.permute}x): edges null mean={null_n.mean():.1f} "
              f"max={null_n.max()} | obs={len(obs)} "
              f"p={(1+(null_n>=len(obs)).sum())/(1+args.permute):.3g}")
        print(f"[*] hub: null mean={null_hub.mean():.1f} max={null_hub.max()} | "
              f"obs={obs_hub} p={(1+(null_hub>=obs_hub).sum())/(1+args.permute):.3g}")

    obs.sort(key=lambda x: -x[3])
    print(f"[✓] {len(obs)} host x parasite edges "
          f"(|cos|>={args.min_sim}, min_lrt={args.min_lrt}, min_cesi={args.min_cesi})")
    for h, p, s, ce in obs[:25]:
        print(f"   host:{h:<5} x par:{p:<5} cos={s:.3f} cesi={ce:.2f}")
    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["host_site", "par_site", "cosine", "cesi"])
            w.writerows(obs)
        print(f"    -> {args.out}")


if __name__ == "__main__":
    main()
