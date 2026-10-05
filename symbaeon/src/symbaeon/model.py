"""
symbaeon/model.py
-----------------
Model-based cross-alignment coselection (the transformer path).

Where `coselect.py` tests *allele-correlation* (do the same individuals carry
non-reference alleles at host site X and parasite site Y — Fisher/CMH on
encoded carrier vectors), this module tests *selection-correlation* using the
PhyloAxialTransformer: it runs the model separately on each MSA (each side
gets its own TN93 distance / positional encoding, so neither side's
substitution rate swamps the other) and asks whether the same individuals /
lineages drive the selection signal in both genes.

One shared per-side model pass feeds two readouts (`--mode`):

  edges   — concentrated selection-correlation. For each selected site we
            build a per-individual "driver" profile by counterfactual
            reversion (delta_lrt: revert one haplotype; delta_lrt_aa: revert
            all carriers of a derived amino acid jointly) and take the
            cross-cosine between host-site and par-site driver profiles.
            Significant excess edges vs a leaf-shuffle null = shared drivers.

  burden  — diffuse selection-correlation. When selection is spread across
            many lineages there is no single driver to correlate, but the
            same individuals may still carry an elevated load of selected
            divergence in both genes. We correlate the LRT-weighted
            attribution burden across individuals, with a variant that
            regresses out relatedness re-estimated from NON-selected sites
            (so the correction removes shared descent, not the signal).

The leaf-axis subtlety: load_alignment_and_tree dedups identical sequences,
collapsing individuals differently per side. We dedup each side ourselves
(unique haplotypes + an individual->haplotype map), run the model on the
uniques, then expand back to per-individual so both sides share the same
individual axis before the cross-individual comparison.
"""

import csv
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pairing import (
    PairedSide,
    build_individual_axis,
    load_manifest,
    load_paired_alignment,
)
from .coselect import load_sample_filter


# ---------------------------------------------------------------------------
# Individual axis + representative sequences
# ---------------------------------------------------------------------------

def _individuals(host_fasta, par_fasta, manifest_path, samples_path):
    """Load both MSAs and return (host, par, shared_individuals)."""
    manifest = load_manifest(manifest_path) if manifest_path else None
    host = load_paired_alignment(host_fasta, "host", manifest)
    par = load_paired_alignment(par_fasta, "parasite", manifest)
    individuals = build_individual_axis(host, par)
    if samples_path:
        keep = load_sample_filter(samples_path)
        individuals = [i for i in individuals if i in keep]
    if len(individuals) < 4:
        raise ValueError("Too few shared individuals between host and parasite MSAs")
    return host, par, individuals


def _representative_seqs(side: PairedSide, individuals: List[str]) -> Dict[str, str]:
    """{individual: representative haplotype} — the first seq_id per individual."""
    return {ind: side.seqs[side.sample_to_seqs[ind][0]] for ind in individuals}


def _load_model(weights=None, variant=None, device=None):
    """Import aeon-core lazily so `symbaeon correlate` works without torch."""
    try:
        from aeon_core.inference import load_model, get_device
    except Exception as e:  # pragma: no cover - depends on optional deps
        raise RuntimeError(f"model-based coselection needs aeon-core + torch: {e}")
    device = device or get_device()
    return load_model(weights=weights, variant=variant, device=device), device


# ---------------------------------------------------------------------------
# Shared per-side model pass
# ---------------------------------------------------------------------------

@dataclass
class SideData:
    """Everything the per-side transformer pass produces, shared by the
    edges and burden readouts so the model only runs once per side."""
    attrs_uniq: np.ndarray      # [L, n_uniq] alpha*delta leaf attributions
    lrts: np.ndarray            # [L] per-site selection LRT
    uniq_seqs: List[str]
    ind2uniq: np.ndarray        # [n_inds] individual -> unique-haplotype index
    taxa_idx: np.ndarray        # [n_leaves] taxon -> unique-haplotype index
    c: object                   # codon tensor
    a: object                   # amino-acid tensor
    d_mat: object
    z: object
    inv_mask: np.ndarray
    taxa: List[str]
    tree_cache: object
    L: int


def _side_base(seqs, inds, model, device, tmpdir, tag) -> SideData:
    """Dedup -> unique haplotypes -> load_alignment_and_tree -> tree_cache ->
    base transformer attributions (alpha*delta) + per-site LRTs."""
    from aeon_core.dataset import load_alignment_and_tree
    from hyphaeon.epistasis import compute_transformer_attributions

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
    attrs = np.asarray(attrs)
    lrts = np.asarray(lrts)

    taxa_idx = np.asarray([int(t.lstrip("u")) for t in taxa])
    attrs_uniq = np.zeros((attrs.shape[0], len(uniq_seqs)), dtype=attrs.dtype)
    for col, uidx in enumerate(taxa_idx):
        attrs_uniq[:, uidx] = attrs[:, col]

    return SideData(attrs_uniq=attrs_uniq, lrts=lrts, uniq_seqs=uniq_seqs,
                    ind2uniq=np.asarray(ind2uniq), taxa_idx=taxa_idx,
                    c=c, a=a, d_mat=d_mat, z=z, inv_mask=inv_mask, taxa=taxa,
                    tree_cache=tree_cache, L=L)


# ---------------------------------------------------------------------------
# Edges readout: counterfactual driver matrix -> cross-cosine
# ---------------------------------------------------------------------------

def _driver_matrix(side: SideData, model, device, method, attr_min_lrt) -> np.ndarray:
    """Per-(site, unique-haplotype) driver matrix [L, n_uniq].

    attn         -> alpha*delta (prediction structure; no counterfactual)
    delta_lrt    -> single-taxon reversion: how much reverting each taxon's
                    non-consensus codon lowers the site LRT
    delta_lrt_aa -> allele-group reversion: revert ALL carriers of each
                    derived amino acid jointly, credit the group's delta-LRT
                    to every carrier (more power for diffuse/shared alleles)
    """
    if method == "attn":
        return side.attrs_uniq
    if method == "delta_lrt":
        from hyphaeon.attribution import attribute_selection
        res = attribute_selection(
            model, side.c, side.a, side.d_mat, side.z, side.inv_mask, side.taxa,
            min_lrt=attr_min_lrt, base_lrts=side.lrts, cache=side.tree_cache)
        out = np.zeros((side.L, len(side.uniq_seqs)), dtype=np.float32)
        for site, meta in res.items():
            for d_sp in meta["driving_species"]:
                out[site, side.taxa_idx[d_sp["taxon_index"]]] = d_sp["delta_lrt"]
        return out
    return _allele_group_drivers(side, model, device, attr_min_lrt)


def _allele_group_drivers(side, model, device, attr_min_lrt) -> np.ndarray:
    """Allele-group counterfactual driver matrix [L, n_uniq]."""
    import torch as _t
    from aeon_core.dataset import CODON_TO_AA, AA_MAP, GENETIC_CODE
    INV = {v: k for k, v in GENETIC_CODE.items()}
    cn = side.c.cpu().numpy()
    an = side.a.cpu().numpy()
    out = np.zeros((side.L, len(side.uniq_seqs)), dtype=np.float32)
    for site in np.where(side.lrts >= attr_min_lrt)[0]:
        site_lrt = float(side.lrts[site])
        codons = cn[site, :, 0]
        aas = an[site, :, 0]
        vals, counts = np.unique(codons, return_counts=True)
        cons_codon = int(vals[np.argmax(counts)])
        cons_aa = AA_MAP.get(CODON_TO_AA.get(INV.get(cons_codon, "NNN"), "-"), 20)
        groups: Dict[int, List[int]] = {}
        for t in np.where(codons != cons_codon)[0]:
            groups.setdefault(int(aas[t]), []).append(int(t))
        for aa_tok, members in groups.items():
            c_mod = cn[site:site + 1].copy()
            a_mod = an[site:site + 1].copy()
            c_mod[0, members, 0] = cons_codon
            a_mod[0, members, 0] = cons_aa
            with _t.no_grad():
                y_mod, _ = model.forward_cached(
                    _t.from_numpy(c_mod).long().to(device),
                    _t.from_numpy(a_mod).long().to(device), side.tree_cache)
            mod_lrt = float(_t.clamp(y_mod.squeeze(-1), min=0.0)
                            .detach().cpu().numpy().ravel()[0])
            for t in members:
                out[site, t] = site_lrt - mod_lrt
    return out


def cross_cosine(Ah: np.ndarray, Ap: np.ndarray) -> np.ndarray:
    """Mean-center each site then cosine between every host-site and
    parasite-site driver profile across individuals -> [L_h, L_p]."""
    Ah = Ah - Ah.mean(axis=1, keepdims=True)
    Ap = Ap - Ap.mean(axis=1, keepdims=True)
    Ah /= np.maximum(np.linalg.norm(Ah, axis=1, keepdims=True), 1e-12)
    Ap /= np.maximum(np.linalg.norm(Ap, axis=1, keepdims=True), 1e-12)
    return Ah @ Ap.T


def edges_from(C, min_sim, lrt_h=None, lrt_p=None, min_lrt=0.0, min_cesi=0.0):
    """(host_site, par_site, cos, cesi) for |cos| >= min_sim AND at least one
    site under selection (lrt >= min_lrt) AND cesi >= min_cesi."""
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


# ---------------------------------------------------------------------------
# Burden readout: per-individual selection burden + neutral-ancestry residual
# ---------------------------------------------------------------------------

def _side_burden(side: SideData, model, device, tmpdir, tag, min_lrt,
                 resid_knn: int = 5):
    """(burden_per_ind, resid_per_ind, n_sel) for one side.

    burden[i] = LRT-weighted attribution mass of i's haplotype over selected
    sites. resid removes the component of burden shared among RELATIVES: each
    individual's burden is predicted from its `resid_knn` nearest neighbours in
    neutral-site TN93 distance space (leave-one-out), so the residual keeps
    only the non-ancestry elevation.

    Why a relatedness kernel and not the earlier MDS+OLS fit: a linear
    regression on low-dim MDS coordinates absorbs ANY group that is linearly
    separable in embedding space — including a scattered, non-ancestry selected
    group that lands in a separable region by chance (sim panel: scattered
    individuals were ~80% predictable from neutral-MDS coords, so OLS erased
    the signal; resid r collapsed to ~0.06). Predicting burden from the
    nearest neighbours in the neutral distance matrix only removes elevation
    that is shared with actual relatives — a clade member's neighbours are its
    high-burden clade-mates (resid -> 0), while a scattered individual's
    neighbours are low-burden background (resid survives). Neighbourhood size
    `resid_knn` must stay below the smallest clade we want to remove, else the
    kernel smooths across the clade boundary and clade signal leaks back in.
    """
    from aeon_core.dataset import compute_tn93_distance_matrix

    sel = side.lrts >= min_lrt
    w = np.clip(side.lrts[sel], 0, None)[:, None]
    burden_uniq = (side.attrs_uniq[sel] * w).sum(axis=0)
    burden = burden_uniq[side.ind2uniq]

    # relatedness from NON-selected sites only — the full alignment bakes the
    # selected-site divergence into "ancestry" and would remove the signal.
    keep = np.where(~sel)[0]
    seq_dict = {
        f"u{i}": "".join(s[3 * j:3 * j + 3] for j in keep)
        for i, s in enumerate(side.uniq_seqs)
    }
    dm_uniq = compute_tn93_distance_matrix(seq_dict, list(seq_dict))
    dm_ind = dm_uniq[np.ix_(side.ind2uniq, side.ind2uniq)].astype(float)
    dm_ind[dm_ind < 0] = np.inf          # tn93 may leave far pairs as -1

    n = len(burden)
    k = max(1, min(resid_knn, n - 1))
    resid = np.empty(n)
    for i in range(n):
        d = dm_ind[i].copy()
        d[i] = np.inf                    # leave-one-out
        nn = np.argsort(d)[:k]
        resid[i] = burden[i] - burden[nn].mean()
    return burden, resid, int(sel.sum())


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float((x @ y) / d) if d > 0 else 0.0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class CoselectResult:
    n_individuals: int
    n_host_sel: int
    n_par_sel: int
    mode: str
    method: str
    # edges readout
    edges: List[Tuple[int, int, float, float]] = field(default_factory=list)
    perm_p_edges: Optional[float] = None
    perm_p_hub: Optional[float] = None
    null_edge_mean: Optional[float] = None
    obs_hub: int = 0
    # burden readout
    r_raw: Optional[float] = None
    p_raw: Optional[float] = None
    r_resid: Optional[float] = None
    p_resid: Optional[float] = None
    null_raw_mean: Optional[float] = None
    null_resid_mean: Optional[float] = None


def run_coselect(
    host_fasta: str,
    par_fasta: str,
    manifest_path: Optional[str] = None,
    samples_path: Optional[str] = None,
    mode: str = "both",
    method: str = "delta_lrt_aa",
    attr_min_lrt: float = 3.84,
    min_sim: float = 0.5,
    min_lrt: float = 1.0,
    min_cesi: float = 0.0,
    permute: int = 0,
    seed: int = 0,
    resid_knn: int = 5,
    weights=None,
    variant=None,
    edges_out: Optional[str] = None,
    burden_out: Optional[str] = None,
    progress: bool = True,
) -> CoselectResult:
    """Model-based selection-correlation. One shared per-side transformer
    pass feeds the edges and/or burden readouts selected by `mode`."""
    host, par, inds = _individuals(host_fasta, par_fasta, manifest_path, samples_path)
    if progress:
        print(f"[*] {len(inds)} shared individuals | "
              f"host seqs={len(host.seqs)} par seqs={len(par.seqs)}")
    model, device = _load_model(weights, variant)

    host_seqs = _representative_seqs(host, inds)
    par_seqs = _representative_seqs(par, inds)
    res = CoselectResult(n_individuals=len(inds), n_host_sel=0, n_par_sel=0,
                         mode=mode, method=method)

    with tempfile.TemporaryDirectory() as td:
        Sh = _side_base(host_seqs, inds, model, device, td, "host")
        Sp = _side_base(par_seqs, inds, model, device, td, "par")
        res.n_host_sel = int((Sh.lrts >= min_lrt).sum())
        res.n_par_sel = int((Sp.lrts >= min_lrt).sum())
        if progress:
            print(f"[*] selected sites (lrt>={min_lrt}): "
                  f"host={res.n_host_sel} par={res.n_par_sel}")

        rng = np.random.default_rng(seed)

        # ---- edges readout (needs the counterfactual driver matrix) ----
        if mode in ("edges", "both"):
            Ah = _driver_matrix(Sh, model, device, method, attr_min_lrt)[:, Sh.ind2uniq]
            Ap = _driver_matrix(Sp, model, device, method, attr_min_lrt)[:, Sp.ind2uniq]
            C = cross_cosine(Ah, Ap)
            obs = edges_from(C, min_sim, Sh.lrts, Sp.lrts, min_lrt, min_cesi)
            res.edges = sorted(obs, key=lambda x: -x[3])
            if permute > 0:
                from collections import Counter
                null_n, null_hub = [], []
                for _ in range(permute):
                    Cp = cross_cosine(Ah, Ap[:, rng.permutation(Ap.shape[1])])
                    e = edges_from(Cp, min_sim, Sh.lrts, Sp.lrts, min_lrt, min_cesi)
                    null_n.append(len(e))
                    null_hub.append(max(Counter(x[1] for x in e).values()) if e else 0)
                null_n, null_hub = np.array(null_n), np.array(null_hub)
                res.obs_hub = max(Counter(x[1] for x in obs).values()) if obs else 0
                res.null_edge_mean = float(null_n.mean())
                res.perm_p_edges = float((1 + (null_n >= len(obs)).sum()) / (1 + permute))
                res.perm_p_hub = float((1 + (null_hub >= res.obs_hub).sum()) / (1 + permute))
            if edges_out:
                with open(edges_out, "w", newline="") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(["host_site", "par_site", "cosine", "cesi"])
                    w.writerows(res.edges)

        # ---- burden readout (base attributions only; no counterfactual) ----
        if mode in ("burden", "both"):
            bh, rh, _ = _side_burden(Sh, model, device, td, "host", min_lrt,
                                     resid_knn)
            bp, rp, _ = _side_burden(Sp, model, device, td, "par", min_lrt,
                                     resid_knn)
            res.r_raw = _pearson(bh, bp)
            res.r_resid = _pearson(rh, rp)
            n_perm = max(permute, 1)
            null_raw = np.empty(n_perm)
            null_resid = np.empty(n_perm)
            for p in range(n_perm):
                perm = rng.permutation(len(inds))
                null_raw[p] = _pearson(bh, bp[perm])
                null_resid[p] = _pearson(rh, rp[perm])
            res.null_raw_mean = float(null_raw.mean())
            res.null_resid_mean = float(null_resid.mean())
            res.p_raw = float((np.sum(null_raw >= res.r_raw) + 1) / (n_perm + 1))
            res.p_resid = float((np.sum(null_resid >= res.r_resid) + 1) / (n_perm + 1))
            if burden_out:
                with open(burden_out, "w", newline="") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(["individual", "burden_host", "burden_par",
                                "resid_host", "resid_par"])
                    for i, ind in enumerate(inds):
                        w.writerow([ind, bh[i], bp[i], rh[i], rp[i]])

    return res
