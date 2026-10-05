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
    hom_only: bool = False,
) -> Tuple[np.ndarray, str]:
    """Encode one site across individuals.

    Returns (values, ref_allele). values is float array with NaN for missing.
    ref_allele is the reference base (from ref_id sequence) or the majority
    allele if no reference is present.

    hom_only (presence encoding only): an individual counts as an alt carrier
    only when alt is its *sole* observed allele. Individuals with a mixed
    allele set {ref, alt} — e.g. a minority clone in a mixed parasite
    infection — are set to NaN (excluded). This removes the detection
    artifact where individuals with more clones are more likely to have at
    least one alt-carrying clone.
    """
    allele_sets = site_allele_sets(side, site, individuals)

    if side.ref_id is not None:
        ref = side.seqs[side.ref_id][site]
        if ref not in side.alphabet:
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
            if hom_only and len(s) > 1:
                continue  # ambiguous mixed/het call -> exclude
            vals[i] = 1.0 if n_alt > 0 else 0.0
        else:  # dosage
            vals[i] = n_alt / len(s)
    return vals, ref


def _variable_sites(
    side: PairedSide,
    individuals: List[str],
    encoding: str,
    min_carriers: int,
    hom_only: bool = False,
) -> List[Tuple[int, np.ndarray, str, int]]:
    """Return [(site, values, ref, n_carriers)] for sites with enough variation.

    A site is kept if at least `min_carriers` individuals carry a
    non-reference allele AND at least `min_carriers` do not (so the
    association test is defined). `hom_only` is passed to _encode_site.
    """
    kept = []
    for site in range(side.length):
        vals, ref = _encode_site(side, site, individuals, encoding, hom_only)
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

def _assoc(
    x: np.ndarray,
    y: np.ndarray,
    encoding: str,
    strata: Optional[np.ndarray] = None,
    corrected: bool = False,
) -> Tuple[float, float]:
    """Association between two encoded site vectors.

    Returns (statistic, p_value). statistic is odds ratio (presence) or
    Spearman rho (dosage). NaN pairs are dropped pairwise.

    If `strata` is given (per-individual stratum labels, e.g. country),
    controls for population structure:
      presence -> Cochran-Mantel-Haenszel common-odds-ratio test
      dosage   -> Spearman on stratum-mean-residualized values

    If `corrected` is True, the vectors are relatedness-residualized
    (continuous), so the discrete tests no longer apply — use a correlation
    test (Pearson; within-stratum ranked Pearson when strata are present).
    """
    ok = ~(np.isnan(x) | np.isnan(y))
    if strata is not None:
        ok &= np.array([s is not None for s in strata])
    x, y = x[ok], y[ok]
    s = strata[ok] if strata is not None else None
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        return np.nan, np.nan

    if corrected:
        if s is not None:
            x = _stratum_ranks(x, s)
            y = _stratum_ranks(y, s)
            if x.std() == 0 or y.std() == 0:
                return np.nan, np.nan
        rho, p = _stats.pearsonr(x, y)
        return float(rho), float(p)

    if encoding == "presence":
        xb, yb = (x > 0), (y > 0)
        if s is None:
            a = int((xb & yb).sum()); b = int((xb & ~yb).sum())
            c = int((~xb & yb).sum()); d = int((~xb & ~yb).sum())
            odds, p = _stats.fisher_exact([[a, b], [c, d]])
            return float(odds), float(p)
        return _cmh_test(xb, yb, s)

    if s is not None:
        # Stratified rank test: rank within each stratum, then correlate the
        # pooled within-stratum quantile ranks. Residualizing raw values and
        # then ranking globally is WRONG for discrete/binary traits — the
        # residualized levels still order by stratum, so global ranks
        # reintroduce the between-stratum structure we meant to remove.
        x = _stratum_ranks(x, s)
        y = _stratum_ranks(y, s)
        if x.std() == 0 or y.std() == 0:
            return np.nan, np.nan
        rho, p = _stats.pearsonr(x, y)
        return float(rho), float(p)
    rho, p = _stats.spearmanr(x, y)
    return float(rho), float(p)


def _cmh_test(xb: np.ndarray, yb: np.ndarray, strata: np.ndarray) -> Tuple[float, float]:
    """Cochran-Mantel-Haenszel test for a common odds ratio across strata.

    xb, yb: boolean carrier vectors; strata: stratum label per observation.
    Returns (common_odds_ratio, p_value). Strata with n<=1 or zero variance
    contribute nothing and are skipped.
    """
    num = den = 0.0          # common OR numerator / denominator
    diff_sum = var_sum = 0.0  # CMH statistic terms
    for stratum in pd_unique(strata):
        m = strata == stratum
        a = float((xb[m] & yb[m]).sum()); b = float((xb[m] & ~yb[m]).sum())
        c = float((~xb[m] & yb[m]).sum()); d = float((~xb[m] & ~yb[m]).sum())
        n = a + b + c + d
        if n <= 1:
            continue
        num += a * d / n
        den += b * c / n
        e_a = (a + b) * (a + c) / n
        v_a = (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
        diff_sum += a - e_a
        var_sum += v_a
    if var_sum <= 0 or den == 0:
        return np.nan, np.nan
    chi2 = diff_sum * diff_sum / var_sum
    p = float(_stats.chi2.sf(chi2, df=1))
    return float(num / den), p


def gene_burden(
    side: PairedSide,
    individuals: List[str],
    encoding: str = "dosage",
    hom_only: bool = False,
    min_carriers: int = 2,
) -> Tuple[np.ndarray, int]:
    """Per-individual gene-level burden score.

    Mean of the per-site encoded values across all variable sites in the
    gene (NaN-aware). Aggregating variants into one score per gene trades
    site resolution for power and a much smaller multiple-testing burden.

      dosage   -> mean alt-allele fraction across variable sites
      presence -> fraction of variable sites where the individual is a carrier
      hom_only -> (presence) mixed/het sites excluded before averaging, so the
                  score is the fraction of *clean* sites that are alt

    Returns (scores, n_sites_used). scores is float array (NaN if no data).
    """
    sites = _variable_sites(side, individuals, encoding, min_carriers, hom_only)
    if not sites:
        return np.full(len(individuals), np.nan), 0
    M = np.column_stack([v for _, v, _, _ in sites])  # individuals x sites
    with np.errstate(invalid="ignore"):
        scores = np.nanmean(M, axis=1)
    return scores, len(sites)


def pd_unique(arr: np.ndarray) -> List:
    """Unique stratum labels preserving order (avoids a pandas dependency)."""
    seen = []
    for v in arr:
        if v not in seen:
            seen.append(v)
    return seen


def _stratum_ranks(x: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Rank x within each stratum, scaled to (0,1) quantiles.

    Ranking within strata before pooling removes between-stratum structure
    in the rank space and keeps strata on a common scale regardless of size.
    """
    out = np.full(len(x), np.nan, dtype=np.float64)
    for stratum in pd_unique(strata):
        m = strata == stratum
        out[m] = _stats.rankdata(x[m]) / (m.sum() + 1)
    return out


# ---------------------------------------------------------------------------
# Relatedness (phylogeny / kinship) correction
# ---------------------------------------------------------------------------
def _distance_matrix(side: PairedSide, individuals: List[str]) -> np.ndarray:
    """Pairwise p-distance between individuals on this side's MSA.

    Uses each individual's first sequence as its representative; distance is
    the fraction of sites where both carry a valid allele and differ. This is
    a relatedness proxy — a phylogeny substitute for the (clonal-ish) parasite
    and a kinship substitute for the host.
    """
    seqs = [side.seqs[side.sample_to_seqs[ind][0]] for ind in individuals]
    charmap = {c: i for i, c in enumerate(sorted(side.alphabet))}
    M = np.array([[charmap.get(c, -1) for c in s] for s in seqs])
    valid = M >= 0
    n = M.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        both = valid[i] & valid[i + 1:]
        comp = both.sum(axis=1)
        diff = ((M[i] != M[i + 1:]) & both).sum(axis=1)
        d = np.where(comp > 0, diff / np.maximum(comp, 1), 0.0)
        D[i, i + 1:] = d
        D[i + 1:, i] = d
    return D


def _mds(D: np.ndarray, k: int) -> np.ndarray:
    """Classical (metric) MDS -> top-k principal coordinates (n x k).

    Eigenvectors of the double-centred squared-distance matrix, scaled by
    sqrt(eigenvalue). Used as relatedness covariates (EIGENSTRAT-style PCs).
    """
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1][:k]
    return evecs[:, idx] * np.sqrt(np.clip(evals[idx], 0, None))


def _residualize(vals: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Regress vals on [1, Z] over observed entries; return residuals.

    NaN entries stay NaN. Removing the relatedness-PC component from a site
    vector leaves the variation not explained by shared descent/ancestry.
    """
    out = np.full(len(vals), np.nan, dtype=np.float64)
    ok = ~np.isnan(vals)
    if ok.sum() <= Z.shape[1] + 1:
        return out
    X = np.column_stack([np.ones(int(ok.sum())), Z[ok]])
    beta, *_ = np.linalg.lstsq(X, vals[ok], rcond=None)
    out[ok] = vals[ok] - X @ beta
    return out


def load_sample_filter(path: str) -> set:
    """Load a set of sample_ids to include.

    Accepts either a TSV with a 'sample_id' or 'ID' column, or a plain
    one-id-per-line list.
    """
    ids = set()
    with open(path, newline="") as fh:
        first = fh.readline()
        fh.seek(0)
        head = first.rstrip("\n").split("\t")
        if len(head) > 1 and ("sample_id" in head or "ID" in head):
            col = "sample_id" if "sample_id" in head else "ID"
            for row in csv.DictReader(fh, delimiter="\t"):
                ids.add(row[col])
        else:
            for line in fh:
                v = line.strip()
                if v:
                    ids.add(v.split("\t")[0])
    return ids


def load_strata(path: str, id_col: Optional[str] = None,
                strata_col: Optional[str] = None) -> Dict[str, str]:
    """Load a stratum mapping TSV -> {sample_id: stratum}.

    id_col defaults to 'sample_id' then 'ID'; strata_col defaults to
    'country' then the second column. Samples with empty/NA strata are
    omitted (treated as unstratified-missing downstream).
    """
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fields = reader.fieldnames or []
        if id_col is None:
            id_col = "sample_id" if "sample_id" in fields else "ID"
        if strata_col is None:
            strata_col = "country" if "country" in fields else (
                [f for f in fields if f != id_col] or [None])[0]
        if id_col not in fields or strata_col not in fields:
            raise ValueError(
                f"Strata file {path}: need columns '{id_col}' and '{strata_col}'; "
                f"found {fields}")
        out = {}
        for row in reader:
            v = row[strata_col].strip()
            if v and v.upper() != "NA":
                out[row[id_col]] = v
    return out


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
class CorrelateResult:
    pairs: List[dict]
    n_individuals: int
    n_host_sites: int
    n_par_sites: int
    n_covariates: int


def run_correlate(
    host_fasta: str,
    par_fasta: str,
    manifest_path: Optional[str] = None,
    host_cov_path: Optional[str] = None,
    strata_path: Optional[str] = None,
    strata_id_col: Optional[str] = None,
    strata_col: Optional[str] = None,
    samples_path: Optional[str] = None,
    encoding: str = "presence",
    hom_only: bool = False,
    min_carriers: int = 10,
    par_pc: int = 0,
    host_pc: int = 0,
    alpha: float = 0.05,
    out_path: Optional[str] = None,
    progress: bool = True,
) -> CorrelateResult:
    """Run cross-alignment coselection between a host and parasite MSA.

    Parameters
    ----------
    host_fasta, par_fasta : host / parasite CDS MSA (FASTA)
    manifest_path         : pairing manifest TSV (seq_id -> sample_id)
    host_cov_path         : optional host covariate TSV (e.g. SV genotypes);
                            each covariate is tested like an extra host site
    strata_path           : optional TSV mapping sample_id -> stratum
                            (e.g. country) for population-structure control
    strata_id_col         : sample-id column in strata file (default: auto)
    strata_col            : stratum column in strata file (default: auto)
    samples_path          : optional file of sample_ids to restrict the
                            individual axis (e.g. one country, cases only)
    encoding              : 'presence' (Fisher/CMH) or 'dosage' (Spearman)
    hom_only              : presence encoding only — count a parasite sample as
                            an alt carrier only when alt is its sole allele;
                            mixed-infection samples are excluded. Removes the
                            minority-clone detection artifact. Host diploid
                            encoding is unaffected (hets are real carriers).
    min_carriers          : min individuals carrying the alt allele to test a site
    par_pc, host_pc       : relatedness correction — number of principal
                            coordinates (from each side's MSA distance matrix)
                            to regress out of that side's site vectors before
                            testing. par_pc controls parasite phylogeny (viral
                            descent); host_pc controls host kinship. 0 = off.
                            When >0 the test is a correlation on residuals
                            (discrete Fisher/CMH no longer applies).
    alpha                 : FDR threshold for reporting
    out_path              : optional TSV of all tested pairs
    """
    manifest = load_manifest(manifest_path) if manifest_path else None
    host = load_paired_alignment(host_fasta, "host", manifest)
    par = load_paired_alignment(par_fasta, "parasite", manifest)
    individuals = build_individual_axis(host, par)
    if samples_path:
        keep = load_sample_filter(samples_path)
        individuals = [i for i in individuals if i in keep]
    if not individuals:
        raise ValueError("No shared sample_ids between host and parasite MSAs")
    if progress:
        print(f"[*] Paired individuals: {len(individuals)} "
              f"(host seqs {len(host.seqs)}, parasite seqs {len(par.seqs)})")

    strata = None
    if strata_path:
        smap = load_strata(strata_path, strata_id_col, strata_col)
        strata = np.array([smap.get(ind) for ind in individuals], dtype=object)
        n_strat = sum(1 for s in strata if s is not None)
        if progress:
            print(f"[*] Strata: {n_strat}/{len(individuals)} individuals "
                  f"assigned ({len(set(s for s in strata if s))} levels)")

    host_sites = _variable_sites(host, individuals, encoding, min_carriers)
    # hom_only applies to the parasite side only: host diploid hets are real
    # carriers, whereas parasite 'hets' are mixed infections where a minority
    # clone would otherwise inflate apparent carriage.
    par_sites = _variable_sites(par, individuals, encoding, min_carriers,
                                hom_only=hom_only)
    covs = []
    if host_cov_path:
        covs = load_host_covariates(host_cov_path, individuals, min_carriers)
    if progress:
        print(f"[*] Variable host sites: {len(host_sites)} "
              f"(+{len(covs)} covariates); variable parasite sites: {len(par_sites)}")

    # Relatedness correction: regress each side's site vectors on that side's
    # top relatedness PCs (distance-matrix MDS). Residualized values are
    # continuous, so the corrected test is a correlation, not Fisher/CMH.
    corrected = par_pc > 0 or host_pc > 0
    Z_par = _mds(_distance_matrix(par, individuals), par_pc) if par_pc > 0 else None
    Z_host = _mds(_distance_matrix(host, individuals), host_pc) if host_pc > 0 else None
    if progress and corrected:
        print(f"[*] Relatedness correction: par_pc={par_pc} host_pc={host_pc}")
    if Z_par is not None:
        par_sites = [(s, _residualize(v, Z_par), r, n) for s, v, r, n in par_sites]

    # host side = MSA sites + covariates (covariates get site label = name)
    host_feats = [(f"site{s+1}", v, r, n) for s, v, r, n in host_sites]
    host_feats += [(f"cov:{name}", v, "-", n) for name, v, n in covs]
    if Z_host is not None:
        host_feats = [(l, _residualize(v, Z_host), r, n) for l, v, r, n in host_feats]

    pairs = []
    stats_arr, p_arr = [], []
    records = []
    for h_label, hv, h_ref, h_n in host_feats:
        for p_site, pv, p_ref, p_n in par_sites:
            stat, p = _assoc(hv, pv, encoding, strata, corrected=corrected)
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

    return CorrelateResult(
        pairs=pairs,
        n_individuals=len(individuals),
        n_host_sites=len(host_sites),
        n_par_sites=len(par_sites),
        n_covariates=len(covs),
    )
