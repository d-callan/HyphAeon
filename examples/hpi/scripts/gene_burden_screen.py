#!/usr/bin/env python
"""
gene_burden_screen.py
---------------------
Gene-level burden coselection screen. Instead of testing every host site x
parasite site, each individual gets one burden score per gene (mean encoded
alt value across that gene's variable sites), then we correlate host-gene
scores against parasite-gene scores across paired individuals.

This collapses the multiple-testing burden (genes x genes, not sites x
sites) and pools rare variants, at the cost of site resolution.

Host features = per-gene burden scores + per-SV carrier status.
Parasite features = per-gene burden scores.

Usage:
  python gene_burden_screen.py --samples gambia_samples.txt \
      --strata samples.tsv --encoding dosage --hom-only --out out.tsv
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy import stats as _stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "symbaeon", "src"))
from symbaeon.pairing import load_manifest, load_paired_alignment
from symbaeon.coselect import (
    gene_burden,
    load_host_covariates,
    load_sample_filter,
    load_strata,
    _stratum_ranks,
    _bh_fdr,
    pd_unique,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SEQDIR = os.path.join(HERE, "..", "results", "paired_seqs")

HOST_GENES = ["GYPA", "GYPB", "GYPE", "CR1", "DARC"]
PAR_GENES = ["EBA175", "EBA140", "EBA181", "PfRH1", "PfRH2a", "PfRH2b", "PfRH4", "PfRH5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqdir", default=SEQDIR)
    ap.add_argument("--manifest", default=os.path.join(SEQDIR, "pairing_manifest.tsv"))
    ap.add_argument("--host-cov", default=os.path.join(SEQDIR, "host_sv_genotypes.tsv"))
    ap.add_argument("--samples", default=None, help="restrict to these sample_ids")
    ap.add_argument("--strata", default=None, help="TSV sample_id->stratum for residualization")
    ap.add_argument("--strata-col", default="country")
    ap.add_argument("--encoding", choices=["presence", "dosage"], default="dosage")
    ap.add_argument("--hom-only", action="store_true")
    ap.add_argument("--min-carriers", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    host_sides = {g: load_paired_alignment(os.path.join(args.seqdir, f"{g}.fasta"), f"host:{g}", manifest)
                  for g in HOST_GENES}
    par_sides = {g: load_paired_alignment(os.path.join(args.seqdir, f"{g}.fasta"), f"par:{g}", manifest)
                 for g in PAR_GENES}

    # common individual axis = intersection across ALL sides
    common = None
    for side in list(host_sides.values()) + list(par_sides.values()):
        s = set(side.sample_to_seqs)
        common = s if common is None else common & s
    individuals = sorted(common)
    if args.samples:
        keep = load_sample_filter(args.samples)
        individuals = [i for i in individuals if i in keep]
    print(f"[*] Individuals: {len(individuals)}")

    strata = None
    if args.strata:
        smap = load_strata(args.strata, strata_col=args.strata_col)
        strata = np.array([smap.get(i) for i in individuals], dtype=object)
        print(f"[*] Strata assigned: {sum(s is not None for s in strata)}/{len(individuals)}")

    # host features: gene burdens + SV covariate carrier status
    host_feats = {}
    for g, side in host_sides.items():
        sc, nsites = gene_burden(side, individuals, args.encoding, args.hom_only, args.min_carriers)
        host_feats[g] = sc
        print(f"  host {g:6s}: {nsites} sites, mean burden {np.nanmean(sc):.4f}")
    if args.host_cov:
        for name, vals, n in load_host_covariates(args.host_cov, individuals, 1):
            host_feats[f"cov:{name}"] = vals
    par_feats = {}
    for g, side in par_sides.items():
        sc, nsites = gene_burden(side, individuals, args.encoding, args.hom_only, args.min_carriers)
        par_feats[g] = sc
        print(f"  par  {g:6s}: {nsites} sites, mean burden {np.nanmean(sc):.4f}")

    # pairwise host x par Spearman (strata-residualized if given)
    records = []
    for hg, hv in host_feats.items():
        for pg, pv in par_feats.items():
            ok = ~(np.isnan(hv) | np.isnan(pv))
            if strata is not None:
                ok &= np.array([s is not None for s in strata])
            x, y = hv[ok], pv[ok]
            s = strata[ok] if strata is not None else None
            if len(x) < 10 or x.std() == 0 or y.std() == 0:
                continue
            if s is not None:
                # stratified rank test: within-stratum quantile ranks + Pearson
                x = _stratum_ranks(x, s)
                y = _stratum_ranks(y, s)
                if x.std() == 0 or y.std() == 0:
                    continue
                rho, p = _stats.pearsonr(x, y)
            else:
                rho, p = _stats.spearmanr(x, y)
            records.append({"host": hg, "par": pg, "n": int(ok.sum()),
                            "rho": float(rho), "p_value": float(p)})

    p_arr = np.array([r["p_value"] for r in records])
    q_arr = _bh_fdr(p_arr)
    for r, q in zip(records, q_arr):
        r["q_value"] = float(q)
        r["significant"] = bool(q <= args.alpha)
    records.sort(key=lambda r: r["p_value"])

    print(f"\n=== gene-burden screen ({args.encoding}{' hom-only' if args.hom_only else ''}"
          f"{', stratified' if args.strata else ''}) ===")
    for r in records:
        flag = " *" if r["significant"] else ""
        print(f"  {r['host']:>14s} x {r['par']:<7s} rho={r['rho']:+.3f} p={r['p_value']:.2e} q={r['q_value']:.3f}{flag}")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(records[0].keys()))
            w.writeheader(); w.writerows(records)
        print(f"[✓] -> {args.out}")


if __name__ == "__main__":
    main()
