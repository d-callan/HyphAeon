"""
symbaeon/screen.py
------------------
Collection-level triage: rank every host_gene x pathogen_gene pair by how
worth-while a full `coselect` deep-dive is.

The screen is cheap because the expensive per-side transformer pass
(`_side_base`) is **per-gene, not per-pair**: for H host x P pathogen genes we
pay H+P model passes once, cache each gene's SideData, then every one of the
H*P pairs is cheap numpy on cached attributions. The counterfactual driver
matrix (`delta_lrt`/`delta_lrt_aa`) — the part that doesn't scale — is deferred
to `coselect` on the shortlisted pairs.

Signals (see SCREENING_PLAN.md):

  per-gene   frac_sel / lrt_mass / burden_var  — "selection activity", NOT a
             substitution-rate proxy (rate conflates neutral drift with
             selection; a housekeeping gene can evolve fast with no
             coevolution-relevant selection).
  per-pair   burden_r_resid (ancestry-corrected diffuse coupling — the primary
             signal) + n_attn_edges (cross-cosine on base attributions — a
             cheap proxy for concentrated co-structure). The RAW burden_r is
             reported for transparency but excluded from the score: it is
             confounded by shared ancestry (clade-driven pairs show high raw r
             with no non-ancestry signal).
"""

import csv
import glob
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .pairing import PairedSide, load_manifest, load_paired_alignment
from .coselect import load_sample_filter
from .model import (
    SideData,
    _load_model,
    _representative_seqs,
    _side_base,
    _side_burden,
    cross_cosine,
    edges_from,
    _pearson,
)


# ---------------------------------------------------------------------------
# Collection loading
# ---------------------------------------------------------------------------

def _load_collection(directory: str, pattern: str, role: str,
                     manifest: Optional[Dict[str, str]]) -> Dict[str, PairedSide]:
    """Load every MSA matching `pattern` in `directory` -> {gene_name: PairedSide}.

    gene_name is the filename stem. All genes share the same individual axis
    downstream (the screen intersects sample_ids across the whole collection).
    """
    genes: Dict[str, PairedSide] = {}
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        name = os.path.splitext(os.path.basename(path))[0]
        genes[name] = load_paired_alignment(path, role, manifest)
    if not genes:
        raise ValueError(f"No {role} MSAs matched {directory}/{pattern}")
    return genes


def _common_axis(genes: List[PairedSide], samples_path: Optional[str]) -> List[str]:
    """Sorted intersection of sample_ids across every gene in the collection."""
    common = set(genes[0].sample_to_seqs)
    for g in genes[1:]:
        common &= set(g.sample_to_seqs)
    inds = sorted(common)
    if samples_path:
        keep = load_sample_filter(samples_path)
        inds = [i for i in inds if i in keep]
    if len(inds) < 4:
        raise ValueError(
            f"Too few shared individuals across the collection ({len(inds)}); "
            "every gene must be aligned over the same samples.")
    return inds


# ---------------------------------------------------------------------------
# Per-gene screen data (one transformer pass, cached)
# ---------------------------------------------------------------------------

@dataclass
class GeneScreen:
    """Cached per-gene model pass + derived screening stats."""
    name: str
    side: SideData
    burden: np.ndarray      # per-individual LRT-weighted burden
    resid: np.ndarray       # burden after regressing out neutral-site ancestry
    attrs_ind: np.ndarray   # base attributions expanded to individuals [L, n_ind]
    n_sel: int
    frac_sel: float
    lrt_mass: float
    burden_var: float


def _gene_screen(name: str, side: PairedSide, inds: List[str], model, device,
                 tmpdir: str, tag: str, min_lrt: float,
                 resid_knn: int) -> GeneScreen:
    """Run the shared per-side pass once and derive the screening stats."""
    seqs = _representative_seqs(side, inds)
    sd = _side_base(seqs, inds, model, device, tmpdir, tag)
    burden, resid, n_sel = _side_burden(sd, model, device, tmpdir, tag, min_lrt,
                                        resid_knn)
    sel = sd.lrts >= min_lrt
    return GeneScreen(
        name=name,
        side=sd,
        burden=burden,
        resid=resid,
        attrs_ind=sd.attrs_uniq[:, sd.ind2uniq],
        n_sel=n_sel,
        frac_sel=float(sel.mean()) if sd.L else 0.0,
        lrt_mass=float(np.clip(sd.lrts[sel], 0, None).sum()),
        burden_var=float(np.var(burden)),
    )


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


@dataclass
class ScreenResult:
    n_individuals: int
    n_host_genes: int
    n_par_genes: int
    rows: List[dict] = field(default_factory=list)   # per-pair, sorted by score
    genes: List[dict] = field(default_factory=list)  # per-gene stats


def run_screen(
    host_dir: str,
    par_dir: str,
    manifest_path: Optional[str] = None,
    samples_path: Optional[str] = None,
    host_pattern: str = "*.fasta",
    par_pattern: str = "*.fasta",
    min_lrt: float = 1.0,
    min_sim: float = 0.5,
    min_cesi: float = 0.0,
    permute: int = 200,
    seed: int = 0,
    resid_knn: int = 5,
    weights=None,
    variant=None,
    top: int = 20,
    out_path: Optional[str] = None,
    progress: bool = True,
) -> ScreenResult:
    """Rank host_gene x pathogen_gene pairs for a `coselect` deep-dive."""
    manifest = load_manifest(manifest_path) if manifest_path else None
    host_genes = _load_collection(host_dir, host_pattern, "host", manifest)
    par_genes = _load_collection(par_dir, par_pattern, "parasite", manifest)
    inds = _common_axis(list(host_genes.values()) + list(par_genes.values()),
                        samples_path)
    if progress:
        print(f"[*] {len(host_genes)} host x {len(par_genes)} pathogen genes | "
              f"{len(inds)} shared individuals")
    model, device = _load_model(weights, variant)

    res = ScreenResult(n_individuals=len(inds),
                       n_host_genes=len(host_genes),
                       n_par_genes=len(par_genes))

    with tempfile.TemporaryDirectory() as td:
        # ---- per-gene passes (the expensive part; once per gene) ----
        H: Dict[str, GeneScreen] = {}
        P: Dict[str, GeneScreen] = {}
        for i, (name, side) in enumerate(host_genes.items()):
            if progress:
                print(f"[*] host gene {i + 1}/{len(host_genes)}: {name}")
            H[name] = _gene_screen(name, side, inds, model, device, td,
                                   f"h_{name}", min_lrt, resid_knn)
        for i, (name, side) in enumerate(par_genes.items()):
            if progress:
                print(f"[*] par  gene {i + 1}/{len(par_genes)}: {name}")
            P[name] = _gene_screen(name, side, inds, model, device, td,
                                   f"p_{name}", min_lrt, resid_knn)

        for g in list(H.values()) + list(P.values()):
            res.genes.append({
                "gene": g.name, "n_sel": g.n_sel, "frac_sel": g.frac_sel,
                "lrt_mass": g.lrt_mass, "burden_var": g.burden_var,
            })

        # ---- per-pair cheap stats on cached SideData ----
        rng = np.random.default_rng(seed)
        n_perm = max(permute, 1)
        rows = []
        for hn, hd in H.items():
            for pn, pd in P.items():
                r_raw = _pearson(hd.burden, pd.burden)
                r_resid = _pearson(hd.resid, pd.resid)
                null_raw = np.empty(n_perm)
                null_resid = np.empty(n_perm)
                for p in range(n_perm):
                    perm = rng.permutation(len(inds))
                    null_raw[p] = _pearson(hd.burden, pd.burden[perm])
                    null_resid[p] = _pearson(hd.resid, pd.resid[perm])
                p_raw = float((np.sum(null_raw >= r_raw) + 1) / (n_perm + 1))
                p_resid = float((np.sum(null_resid >= r_resid) + 1) / (n_perm + 1))

                C = cross_cosine(hd.attrs_ind, pd.attrs_ind)
                n_attn = len(edges_from(C, min_sim, hd.side.lrts,
                                        pd.side.lrts, min_lrt, min_cesi))
                # density = fraction of the L_h x L_p site-pair space that is
                # correlated. Raw count scales with gene length, so long genes
                # would dominate regardless of coupling; density is scale-free.
                denom = max(hd.side.L * pd.side.L, 1)
                edge_density = n_attn / denom

                rows.append({
                    "host_gene": hn, "par_gene": pn,
                    "frac_sel_h": hd.frac_sel, "frac_sel_p": pd.frac_sel,
                    "lrt_mass_h": hd.lrt_mass, "lrt_mass_p": pd.lrt_mass,
                    "burden_var_h": hd.burden_var, "burden_var_p": pd.burden_var,
                    "burden_r": r_raw, "p_raw": p_raw,
                    "burden_r_resid": r_resid, "p_resid": p_resid,
                    "n_attn_edges": n_attn, "edge_density": edge_density,
                })

        # ---- composite score (z-scored; min() so BOTH partners must be active)
        zr = _z([r["burden_r_resid"] for r in rows])
        ze = _z([r["edge_density"] for r in rows])
        zf = _z([min(r["frac_sel_h"], r["frac_sel_p"]) for r in rows])
        zl = _z([min(r["lrt_mass_h"], r["lrt_mass_p"]) for r in rows])
        for r, a, b, c, d in zip(rows, zr, ze, zf, zl):
            r["score"] = float(a + b + c + d)
        rows.sort(key=lambda r: -r["score"])
        res.rows = rows

    if out_path:
        cols = ["host_gene", "par_gene", "score", "burden_r_resid", "p_resid",
                "burden_r", "p_raw", "n_attn_edges", "edge_density",
                "frac_sel_h", "frac_sel_p", "lrt_mass_h", "lrt_mass_p",
                "burden_var_h", "burden_var_p"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in cols})
        if progress:
            print(f"[*] wrote {out_path}")
    return res
