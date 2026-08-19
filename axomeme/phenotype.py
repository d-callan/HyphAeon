"""
axomeme/phenotype.py
--------------------
Directional Phenotype-Genotype Association Mapping (PhyloWAS) and
Phenotype-Associated Residue Signature (PARS) extraction.
"""

import os
import re
import sys
import json
import fnmatch
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
from scipy.stats import poisson, norm

from .dataset import (
    AA_MAP,
    CODON_TO_AA,
    parse_alignment_sequences
)

REV_AA_MAP = {v: k for k, v in AA_MAP.items()}

PRESETS = {
    "echolocation": {
        "title": "Mammalian Echolocation Convergence",
        "description": "Microchiropteran bats and odontocete toothed whales.",
        "foreground": [
            "rhi*", "hip*", "myo*", "pte*", "mor*", "min*", "emb*", "cra*", "meg*", "mol*",
            "turTru", "delLeu", "orcOrc", "gloMel", "phaSin", "graGri", "neoPho", "phoPho",
            "phyCat", "kogBre", "kogSim", "mesBid", "zipCav", "plaGan", "iniGeo", "lipVex", "ponBla"
        ],
        "controls": ["pteAle", "pteRod", "pteRuf", "pteGig", "pteVam", "ptePse", "bal*", "megNov", "eubGla", "escRob"]
    },
    "marine": {
        "title": "Marine Mammal Transition & Deep Diving Hypoxia",
        "description": "Cetaceans, Pinnipeds, Sirenians, and Sea Otters.",
        "foreground": [
            "enhLut*", "pusHis*", "pusSib*", "halGryp*", "phoVit*", "phoLar*", "eriBar*", "neoSch*",
            "odoRos*", "zalCal*", "eumJub*", "arcAus*", "arcGaz*", "otoFla*", "mirLeo*", "mirAng*",
            "lepWed*", "hydLep*", "lobCar*", "ommRos*", "triMan*", "triSen*", "triInu*", "dugDug*",
            "turTru*", "delLeu*", "orcOrc*", "gloMel*", "phaSin*", "graGri*", "neoPho*", "phoPho*",
            "balMys*", "balAcu*", "balPhy*", "balMus*", "megNov*", "eubGla*", "escRob*", "phyCat*",
            "kogBre*", "kogSim*", "mesBid*", "zipCav*", "plaGan*", "iniGeo*", "lipVex*", "ponBla*"
        ]
    },
    "fossorial": {
        "title": "Subterranean Fossoriality & Hypercapnic Hypoxia",
        "description": "Naked mole-rats, blind mole-rats, golden moles, star-nosed moles, pocket gophers.",
        "foreground": [
            "hetGla*", "fukDam*", "cryAns*", "nanGal*", "nanEhr*", "spaCar*", "conCri*", "talEur*",
            "talOcc*", "scaMos*", "scaAqu*", "chrAsi*", "chrSta*", "uroGra*", "geoBur*", "thoTal*",
            "canTub*", "ellLut*", "ellTal*"
        ]
    },
    "hibernation": {
        "title": "True Hibernation & Metabolic Torpor",
        "description": "Marmots, ground squirrels, dormice, tenrecs, hedgehogs, Myotis bats.",
        "foreground": [
            "ictTri*", "uroPar*", "speCit*", "speDau*", "marFla*", "marMar*", "marVan*", "marMon*",
            "gliGli*", "dryNit*", "musAve*", "eriEur*", "tenEca*", "echTel*", "micTal*", "myoLuc*",
            "myoDau*", "myoMyo*", "myoNat*", "myoBra*", "ursArc*"
        ]
    },
    "longevity": {
        "title": "Extreme Longevity & Peto's Paradox Centenarians",
        "description": "Bowhead whale, naked mole-rat, Brandt's bat, elephants, humans.",
        "foreground": [
            "balMys*", "hetGla*", "myoBra*", "loxAfr*", "eleMax*", "homSap*"
        ]
    },
    "high_altitude": {
        "title": "High-Altitude Hypoxia Adaptation",
        "description": "Yak, Tibetan antelope, snow leopard, vicuna, pikas, chinchilla.",
        "foreground": [
            "bosGru*", "bosMut*", "panHod*", "panUnc*", "vicVic*", "vicPac*", "chiLan*", "ochCur*",
            "ochPri*", "ochArg*"
        ]
    },
    "cardenolide": {
        "title": "Insect Cardenolide Resistance (ATP1a)",
        "description": "Chrysochus, Tetraopes, Danaus (Monarch), Oncopeltus.",
        "foreground": [
            "chrysochus*", "tetraopes*", "danaus*", "oncopeltus*", "chrysomela*"
        ]
    },
    "dim_light": {
        "title": "Low-Light & Deep-Sea Rhodopsin Vision",
        "description": "Deep-sea teleosts, cavefish, coelacanth, marine diving mammals.",
        "foreground": [
            "*eel*", "*conger*", "*scabbard*", "*blackdragon*", "*viperfish*", "*loosejaw*",
            "*lampfish*", "*thornyhead*", "*cavefish*", "*dolphin*", "*coelacanth*"
        ]
    }
}

def resolve_phenotype_vector(
    taxa: List[str],
    preset: Optional[str] = None,
    foreground: Optional[Union[str, List[str]]] = None,
    background: Optional[Union[str, List[str]]] = None,
    phenotype_file: Optional[str] = None,
    trait_col: Optional[str] = None,
    species_col: Optional[str] = None,
    continuous: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Constructs the phenotypic trait vector y in R^N across all N taxa.
    Supports flexible presets, metadata table parsing, and inline pattern matching.
    """
    N = len(taxa)
    y = np.zeros(N, dtype=float)
    meta = {
        "mode": "discrete",
        "foreground_count": 0,
        "background_count": 0,
        "description": ""
    }

    # 1. Phenotype Metadata File
    if phenotype_file:
        if not os.path.exists(phenotype_file):
            raise FileNotFoundError(f"Phenotype file not found: {phenotype_file}")

        sep = "\t" if phenotype_file.endswith((".tsv", ".tab")) else ","
        df_pheno = pd.read_csv(phenotype_file, sep=sep)

        # Auto-detect species column if not specified
        if not species_col:
            for c in ["species", "taxon", "taxa", "tree_leaf_name", "assembly", "id", "name", "species_name"]:
                match = [col for col in df_pheno.columns if col.lower() == c]
                if match:
                    species_col = match[0]
                    break
            if not species_col:
                species_col = df_pheno.columns[0]

        # Auto-detect trait column if not specified
        if not trait_col:
            cand_cols = [c for c in df_pheno.columns if c != species_col]
            if not cand_cols:
                raise ValueError(f"No valid trait column found in {phenotype_file}")
            trait_col = cand_cols[0]

        trait_dict = {}
        for _, row in df_pheno.iterrows():
            sp = str(row[species_col]).strip()
            val = row[trait_col]
            trait_dict[sp] = val
            trait_dict[sp.lower()] = val

        # Match taxa against dictionary
        matched_values = []
        for i, t in enumerate(taxa):
            val = None
            if t in trait_dict:
                val = trait_dict[t]
            elif t.lower() in trait_dict:
                val = trait_dict[t.lower()]
            else:
                # Substring / pattern matching
                for k, v in trait_dict.items():
                    if k in t or t in k:
                        val = v
                        break

            if val is not None:
                try:
                    num_val = float(val)
                    y[i] = num_val
                    matched_values.append(num_val)
                except ValueError:
                    # String category
                    y[i] = 1.0 if str(val).lower() in ["1", "true", "yes", "case", "foreground", "target", "positive"] else 0.0
                    matched_values.append(y[i])

        if continuous:
            meta["mode"] = "continuous"
            if len(matched_values) > 0 and np.std(matched_values) > 0:
                y = (y - np.mean(y)) / np.std(y)
            meta["description"] = f"Continuous trait '{trait_col}' from {os.path.basename(phenotype_file)}"
        else:
            meta["mode"] = "discrete"
            meta["foreground_count"] = int(np.sum(y > 0))
            meta["background_count"] = int(np.sum(y <= 0))
            meta["description"] = f"Discrete trait '{trait_col}' from {os.path.basename(phenotype_file)}"

        return y, meta

    # 2. Curated Presets
    if preset:
        preset_key = preset.lower().replace("-", "_").strip()
        if preset_key not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
        p_info = PRESETS[preset_key]
        patterns = p_info["foreground"]
        meta["description"] = f"{p_info['title']} ({p_info['description']})"
        for i, t in enumerate(taxa):
            for pat in patterns:
                if fnmatch.fnmatch(t.lower(), pat.lower()) or (pat.lower() in t.lower()):
                    y[i] = 1.0
                    break

        meta["mode"] = "discrete"
        meta["foreground_count"] = int(np.sum(y > 0))
        meta["background_count"] = int(np.sum(y <= 0))
        return y, meta

    # 3. Inline Foreground / Background Patterns
    if foreground:
        if isinstance(foreground, str):
            fg_list = [p.strip() for p in foreground.split(",") if p.strip()]
        else:
            fg_list = foreground

        for i, t in enumerate(taxa):
            for pat in fg_list:
                if fnmatch.fnmatch(t.lower(), pat.lower()) or (pat.lower() in t.lower()):
                    y[i] = 1.0
                    break

        meta["mode"] = "discrete"
        meta["foreground_count"] = int(np.sum(y > 0))
        meta["background_count"] = int(np.sum(y <= 0))
        meta["description"] = f"User-specified foreground patterns: {fg_list}"
        return y, meta

    raise ValueError("Must provide one of --preset, --phenotype-file, or --foreground.")

def run_phenotype_association(
    alignment_path: str,
    preset: Optional[str] = None,
    foreground: Optional[Union[str, List[str]]] = None,
    background: Optional[Union[str, List[str]]] = None,
    phenotype_file: Optional[str] = None,
    trait_col: Optional[str] = None,
    species_col: Optional[str] = None,
    continuous: bool = False,
    min_taxa_per_site: int = 10,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Executes directional Phenotype-Genotype association (PhyloWAS) on a codon alignment.
    Strictly enforces the consensus non-gap reference invariant and sequenced-taxa null scaling.
    """
    seq_dict = parse_alignment_sequences(alignment_path)
    if not seq_dict:
        raise ValueError(f"Could not parse sequences from {alignment_path}")

    taxa = list(seq_dict.keys())
    seqs = list(seq_dict.values())
    N = len(taxa)
    L = len(seqs[0]) // 3

    y, meta = resolve_phenotype_vector(
        taxa=taxa,
        preset=preset,
        foreground=foreground,
        background=background,
        phenotype_file=phenotype_file,
        trait_col=trait_col,
        species_col=species_col,
        continuous=continuous
    )

    is_fg = (y > 0)
    fg_count = int(np.sum(is_fg))
    if fg_count < 2 and not continuous:
        raise ValueError(f"Insufficient foreground taxa ({fg_count}) matching criteria among {N} taxa.")

    # Codon to Amino Acid Tokenization
    aa_mat = np.zeros((N, L), dtype=int)
    for i in range(N):
        s_seq = seqs[i]
        for s in range(L):
            codon = s_seq[s*3:s*3+3].upper()
            aa = CODON_TO_AA.get(codon, '-')
            aa_mat[i, s] = AA_MAP.get(aa, 20)

    # Consensus Reference Invariant strictly over sequenced taxa
    bg_consensus = np.zeros(L, dtype=int)
    for s in range(L):
        valid_bg = aa_mat[~is_fg, s][aa_mat[~is_fg, s] < 20] if fg_count > 0 else aa_mat[:, s][aa_mat[:, s] < 20]
        bg_consensus[s] = np.argmax(np.bincount(valid_bg)) if len(valid_bg) > 0 else 20

    # Substitution Matrix B [L, N]
    B = np.zeros((L, N), dtype=float)
    for s in range(L):
        if bg_consensus[s] < 20:
            for i in range(N):
                if aa_mat[i, s] < 20 and aa_mat[i, s] != bg_consensus[s]:
                    B[s, i] = 1.0

    # Unit Hypersphere Projection
    y_norm = y / np.linalg.norm(y) if np.linalg.norm(y) > 0 else y
    proj = B @ y_norm
    spectral_energy = float(np.linalg.norm(proj))
    frob_norm = float(np.linalg.norm(B))
    norm_spectral_ratio = spectral_energy / frob_norm if frob_norm > 0 else 0.0

    # Site-Level Directional Associations
    site_results = []
    for s in range(L):
        if bg_consensus[s] < 20:
            valid_mask = (aa_mat[:, s] < 20)
            tot_mut = int(np.sum(B[s, :]))
            if np.sum(valid_mask) >= min_taxa_per_site and tot_mut > 0:
                b_sub = B[s, valid_mask]
                y_sub = y[valid_mask]
                b_norm = b_sub / np.linalg.norm(b_sub) if np.linalg.norm(b_sub) > 0 else b_sub
                y_sub_norm = y_sub / np.linalg.norm(y_sub) if np.linalg.norm(y_sub) > 0 else y_sub
                assoc = float(b_norm @ y_sub_norm)

                shared = int(np.sum((B[s, :] > 0) & is_fg))
                N_valid = int(np.sum(valid_mask))
                N_fg_valid = int(np.sum(valid_mask & is_fg))
                exp_shared = (tot_mut * N_fg_valid) / N_valid if N_valid > 0 else 0.0
                p_val = 1.0 - poisson.cdf(shared - 1, exp_shared) if exp_shared > 0 else 1.0

                ref_aa = REV_AA_MAP.get(bg_consensus[s], '-')
                fg_valid_aa = aa_mat[is_fg, s][aa_mat[is_fg, s] < 20]
                derived_aa = REV_AA_MAP.get(np.argmax(np.bincount(fg_valid_aa)), '-') if len(fg_valid_aa) > 0 else ref_aa

                fg_freq = (np.sum(fg_valid_aa == AA_MAP.get(derived_aa, 20)) / len(fg_valid_aa) * 100.0) if len(fg_valid_aa) > 0 else 0.0
                bg_valid_aa = aa_mat[~is_fg, s][aa_mat[~is_fg, s] < 20]
                bg_freq = (np.sum(bg_valid_aa == AA_MAP.get(derived_aa, 20)) / len(bg_valid_aa) * 100.0) if len(bg_valid_aa) > 0 else 0.0

                site_results.append({
                    "site": s + 1,
                    "ref_aa": ref_aa,
                    "derived_aa": derived_aa,
                    "total_mutations": tot_mut,
                    "shared_foreground_mutations": shared,
                    "expected_shared": float(exp_shared),
                    "association_rho": float(assoc),
                    "p_value": float(p_val),
                    "foreground_freq_pct": float(fg_freq),
                    "background_freq_pct": float(bg_freq)
                })

    site_results.sort(key=lambda x: x["association_rho"], reverse=True)

    # Benjamini-Hochberg FDR
    m = len(site_results)
    if m > 0:
        p_sorted_idx = np.argsort([x["p_value"] for x in site_results])
        min_q = 1.0
        for rank, idx in reversed(list(enumerate(p_sorted_idx))):
            p_val = site_results[idx]["p_value"]
            q_val = (p_val * m) / (rank + 1)
            if q_val < min_q:
                min_q = q_val
            site_results[idx]["q_value"] = min(min_q, 1.0)
    
    # Dual-Track Extreme-Value Statistics
    max_assoc = float(site_results[0]["association_rho"]) if site_results else 0.0
    # Null std dev for correlation ~ 1 / sqrt(N_valid)
    sigma_null = 1.0 / np.sqrt(max(10, N))
    z_single = max_assoc / sigma_null if sigma_null > 0 else 0.0
    p_single = float(2.0 * norm.sf(abs(z_single)))
    p_single = max(1e-15, min(1.0, p_single))
    
    # Gumbel EVD across L sites
    p_evd = float(-np.expm1(-L * p_single))
    p_evd = max(1e-15, min(1.0, p_evd))
    
    score_track_a = float(-np.log10(p_evd))
    score_track_b = float(norm_spectral_ratio)
    dual_track_composite = float(max(score_track_a / 10.0, score_track_b))

    # Extract PARS Signature Bracket
    top_pars_sites = [f"{x['ref_aa']}{x['site']}{x['derived_aa']}" for x in site_results if x["association_rho"] >= 0.50][:15]
    compact_pars = f"[ {' - '.join(top_pars_sites)} ]" if top_pars_sites else "[]"

    return {
        "alignment": alignment_path,
        "taxa_count": N,
        "codon_count": L,
        "phenotype_meta": meta,
        "spectral_energy": spectral_energy,
        "norm_spectral_ratio": norm_spectral_ratio,
        "max_assoc": max_assoc,
        "p_evd_length_adjusted": p_evd,
        "score_track_a": score_track_a,
        "score_track_b": score_track_b,
        "dual_track_composite": dual_track_composite,
        "compact_pars_signature": compact_pars,
        "significant_sites_count": len([x for x in site_results if x.get("q_value", 1.0) <= alpha]),
        "sites": site_results
    }
