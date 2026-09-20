"""
symbaeon/coselect.py
--------------------
Cross-alignment coselection: site-site association between a host MSA and a
parasite MSA computed across paired *individuals* (not across shared tree
leaves as in HyphAeon's within-alignment coselection).

Encodings (per individual, per site):
  presence : 1 if any of the individual's sequences carries a non-reference
             allele, else 0. Robust to unphased diploidy / mixed infection.
  dosage   : fraction of the individual's sequences carrying a non-reference
             allele (0, 0.5, 1 for diploid; 0/1 for haploid clones).

Association statistics:
  presence -> Fisher exact test on the 2x2 carrier table (+ odds ratio)
  dosage   -> Spearman rank correlation on dosage vectors (+ rho)

Multiple testing: Benjamini-Hochberg FDR across all tested site pairs.
"""

import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as _stats

from .pairing import (
    PairedSide,
    build_individual_axis,
    load_manifest,
    load_paired_alignment,
    site_allele_sets,
)

VALID_BASES = frozenset("ACGT")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_site(
    side: PairedSide,
    site: int,
    individuals: List[str],
    encoding: str,
) -> Tuple[np.ndarray, str]:
    """Encode one site across individuals.

    Returns (values, ref_allele). values is float array with NaN for missing.
    ref_allele is the reference base (from ref_id sequence) or the majority
    allele if no reference is present.
    """
    allele_sets = site_allele_sets(side, site, individuals)

    if side.ref_id is not None:
        ref = side.seqs[side.ref_id][site]
        if ref not in VALID_BASES:
            ref = None
    else:
        ref = None
    if ref is None:
        # majority allele across all observed alleles
        counts: Dict[str, int] = {}
        for s in allele_sets:
            for a in s:
                counts[a] = counts.get(a, 0) + 1
        ref = max(counts, key=counts.get) if counts else "N"

    vals = np.full(len(individuals), np.nan, dtype=np.float64)
    for i, s in enumerate(allele_sets):
        if not s:
            continue  # fully missing
        n_alt = sum(1 for a in s if a != ref)
        if encoding == "presence":
            vals[i] = 1.0 if n_alt > 0 else 0.0
        else:  # dosage
            vals[i] = n_alt / len(s)
    return vals, ref


def _variable_sites(
    side: PairedSide,
    individuals: List[str],
    encoding: str,
    min_carriers: int,
) -> List[Tuple[int, np.ndarray, str, int]]:
    """Return [(site, values, ref, n_carriers)] for sites with enough variation.

    A site is kept if at least `min_carriers` individuals carry a
    non-reference allele AND at least `min_carriers` do not (so the
    association test is defined).
    """
    kept = []
    for site in range(side.length):
        vals, ref = _encode_site(side, site, individuals, encoding)
        ok = ~np.isnan(vals)
        n_obs = int(ok.sum())
        if n_obs < 2 * min_carriers:
            continue
        n_alt = int((vals[ok] > 0).sum())
        n_ref = n_obs - n_alt
        if n_alt >= min_carriers and n_ref >= min_carriers:
            kept.append((site, vals, ref, n_alt))
    return kept


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------

def _assoc(x: np.ndarray, y: np.ndarray, encoding: str) -> Tuple[float, float]:
    """Association between two encoded site vectors.

    Returns (statistic, p_value). statistic is odds ratio (presence) or
    Spearman rho (dosage). NaN pairs are dropped pairwise.
    """
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        return np.nan, np.nan

    if encoding == "presence":
        a = int(((x > 0) & (y > 0)).sum())
        b = int(((x > 0) & (y == 0)).sum())
        c = int(((x == 0) & (y > 0)).sum())
        d = int(((x == 0) & (y == 0)).sum())
        odds, p = _stats.fisher_exact([[a, b], [c, d]])
        return float(odds), float(p)
    rho, p = _stats.spearmanr(x, y)
    return float(rho), float(p)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaN p-values -> NaN q."""
    q = np.full_like(pvals, np.nan, dtype=np.float64)
    ok = ~np.isnan(pvals)
    p = pvals[ok]
    m = len(p)
    if m == 0:
        return q
    order = np.argsort(p)
    ranked = p[order]
    qv = ranked * m / (np.arange(m) + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(qv, 0, 1)
    q[ok] = out
    return q


# ---------------------------------------------------------------------------
# Host covariates (e.g. glycophorin SVs)
# ---------------------------------------------------------------------------

def load_host_covariates(
    path: str,
    individuals: List[str],
    min_carriers: int,
) -> List[Tuple[str, np.ndarray, int]]:
    """Load a host covariate TSV (sample_id + one column per covariate).

    Each covariate is encoded as carrier status: any non-reference allele
    counts as a carrier. The reference allele is taken as the *major* allele
    across samples (so 'DD'/'II' style genotype strings work regardless of
    which allele is ancestral). Returns [(name, values, n_carriers)].
    """
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if "sample_id" not in reader.fieldnames:
            raise ValueError(f"Covariate file {path} needs a 'sample_id' column")
        cols = [c for c in reader.fieldnames if c != "sample_id"]
        rows = {r["sample_id"]: r for r in reader}

    out = []
    for col in cols:
        # collect per-sample genotype strings
        genos = [rows[ind][col] if ind in rows else "" for ind in individuals]
        # major allele = reference
        allele_counts: Dict[str, int] = {}
        for g in genos:
            for ch in g:
                if ch and ch != "N":
                    allele_counts[ch] = allele_counts.get(ch, 0) + 1
        if not allele_counts:
            continue
        ref = max(allele_counts, key=allele_counts.get)
        vals = np.full(len(individuals), np.nan, dtype=np.float64)
        for i, g in enumerate(genos):
            if not g or g == "NA":
                continue
            vals[i] = 1.0 if any(ch != ref and ch != "N" for ch in g) else 0.0
        ok = ~np.isnan(vals)
        n_alt = int((vals[ok] > 0).sum())
        n_ref = int(ok.sum()) - n_alt
        if n_alt >= min_carriers and n_ref >= min_carriers:
            out.append((col, vals, n_alt))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class CoselectResult:
    pairs: List[dict]
    n_individuals: int
    n_host_sites: int
    n_par_sites: int
    n_covariates: int


def run_coselect(
    host_fasta: str,
    par_fasta: str,
    manifest_path: Optional[str] = None,
    host_cov_path: Optional[str] = None,
    encoding: str = "presence",
    min_carriers: int = 10,
    alpha: float = 0.05,
    out_path: Optional[str] = None,
    progress: bool = True,
) -> CoselectResult:
    """Run cross-alignment coselection between a host and parasite MSA.

    Parameters
    ----------
    host_fasta, par_fasta : host / parasite CDS MSA (FASTA)
    manifest_path         : pairing manifest TSV (seq_id -> sample_id)
    host_cov_path         : optional host covariate TSV (e.g. SV genotypes);
                            each covariate is tested like an extra host site
    encoding              : 'presence' (Fisher) or 'dosage' (Spearman)
    min_carriers          : min individuals carrying the alt allele to test a site
    alpha                 : FDR threshold for reporting
    out_path              : optional TSV of all tested pairs
    """
    manifest = load_manifest(manifest_path) if manifest_path else None
    host = load_paired_alignment(host_fasta, "host", manifest)
    par = load_paired_alignment(par_fasta, "parasite", manifest)
    individuals = build_individual_axis(host, par)
    if not individuals:
        raise ValueError("No shared sample_ids between host and parasite MSAs")
    if progress:
        print(f"[*] Paired individuals: {len(individuals)} "
              f"(host seqs {len(host.seqs)}, parasite seqs {len(par.seqs)})")

    host_sites = _variable_sites(host, individuals, encoding, min_carriers)
    par_sites = _variable_sites(par, individuals, encoding, min_carriers)
    covs = []
    if host_cov_path:
        covs = load_host_covariates(host_cov_path, individuals, min_carriers)
    if progress:
        print(f"[*] Variable host sites: {len(host_sites)} "
              f"(+{len(covs)} covariates); variable parasite sites: {len(par_sites)}")

    # host side = MSA sites + covariates (covariates get site label = name)
    host_feats = [(f"site{s+1}", v, r, n) for s, v, r, n in host_sites]
    host_feats += [(f"cov:{name}", v, "-", n) for name, v, n in covs]

    pairs = []
    stats_arr, p_arr = [], []
    records = []
    for h_label, hv, h_ref, h_n in host_feats:
        for p_site, pv, p_ref, p_n in par_sites:
            stat, p = _assoc(hv, pv, encoding)
            records.append({
                "host_feature": h_label,
                "host_ref": h_ref,
                "host_carriers": h_n,
                "par_site": p_site + 1,
                "par_ref": p_ref,
                "par_carriers": p_n,
                "stat": stat,
                "p_value": p,
            })
            stats_arr.append(stat)
            p_arr.append(p)

    p_arr = np.array(p_arr, dtype=np.float64)
    q_arr = _bh_fdr(p_arr)
    for rec, q in zip(records, q_arr):
        rec["q_value"] = float(q) if not np.isnan(q) else np.nan
        rec["significant"] = bool(not np.isnan(q) and q <= alpha)
    records.sort(key=lambda r: (np.isnan(r["q_value"]), r["q_value"]))
    pairs = records

    if out_path:
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(records[0].keys()) if records else
                               ["host_feature", "host_ref", "host_carriers", "par_site",
                                "par_ref", "par_carriers", "stat", "p_value", "q_value", "significant"])
            w.writeheader()
            w.writerows(records)
        if progress:
            n_sig = sum(r["significant"] for r in records)
            print(f"[✓] {len(records)} pairs tested, {n_sig} significant (q<={alpha}) -> {out_path}")

    return CoselectResult(
        pairs=pairs,
        n_individuals=len(individuals),
        n_host_sites=len(host_sites),
        n_par_sites=len(par_sites),
        n_covariates=len(covs),
    )
