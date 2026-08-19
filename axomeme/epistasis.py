"""
axomeme/epistasis.py
--------------------
Multi-Scale Epistatic Sector Mining (ESSM) and Co-Selection Network Engine
implementing the Two-Stage Seed-and-Extend (TSE) Algorithm.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import poisson

from .dataset import (
    AA_MAP,
    CODON_TO_AA,
    parse_alignment_sequences
)

REV_AA_MAP = {v: k for k, v in AA_MAP.items()}

def run_epistatic_sector_mining(
    alignment_path: str,
    tree_path: Optional[str] = None,
    min_clique_size: int = 3,
    min_sim: float = 0.50,
    max_p_pair: float = 0.05,
    min_mutations: int = 2,
    max_sectors: int = 15
) -> Dict[str, Any]:
    """
    Mines cooperative allosteric sectors and co-selection networks from a codon alignment.
    Implements Theorem 1 and Algorithm 1 (Two-Stage Seed-and-Extend TSE).
    """
    seq_dict = parse_alignment_sequences(alignment_path)
    if not seq_dict:
        raise ValueError(f"Could not parse sequences from {alignment_path}")

    taxa = list(seq_dict.keys())
    seqs = list(seq_dict.values())
    N = len(taxa)
    L = len(seqs[0]) // 3

    # Codon to Amino Acid Tokenization
    aa_mat = np.zeros((N, L), dtype=int)
    for i in range(N):
        s_seq = seqs[i]
        for s in range(L):
            codon = s_seq[s*3:s*3+3].upper()
            aa = CODON_TO_AA.get(codon, '-')
            aa_mat[i, s] = AA_MAP.get(aa, 20)

    # Dynamic Consensus Reference strictly over sequenced taxa
    consensus = np.zeros(L, dtype=int)
    for s in range(L):
        valid = aa_mat[:, s][aa_mat[:, s] < 20]
        consensus[s] = np.argmax(np.bincount(valid)) if len(valid) > 0 else 20

    # Substitution Matrix B [L, N]
    B = np.zeros((L, N), dtype=float)
    for s in range(L):
        if consensus[s] < 20:
            for i in range(N):
                if aa_mat[i, s] < 20 and aa_mat[i, s] != consensus[s]:
                    B[s, i] = 1.0

    # Filter active polymorphic sites
    active_sites = [s for s in range(L) if np.sum(B[s, :]) >= min_mutations]
    if len(active_sites) < min_clique_size:
        return {
            "alignment": alignment_path,
            "taxa_count": N,
            "codon_count": L,
            "active_sites_count": len(active_sites),
            "sectors": [],
            "edges": []
        }

    # Build Co-Selection Graph G = (V, E)
    G = nx.Graph()
    for s in active_sites:
        G.add_node(s + 1, ref=REV_AA_MAP.get(consensus[s], '-'))

    edge_list = []
    for i_idx, s1 in enumerate(active_sites):
        b1 = B[s1, :]
        n1 = np.linalg.norm(b1)
        if n1 == 0:
            continue
        for s2 in active_sites[i_idx + 1:]:
            b2 = B[s2, :]
            n2 = np.linalg.norm(b2)
            if n2 == 0:
                continue

            sim = float((b1 @ b2) / (n1 * n2))
            shared = int(np.sum((b1 > 0) & (b2 > 0)))
            k1, k2 = int(np.sum(b1 > 0)), int(np.sum(b2 > 0))
            exp_shared = (k1 * k2) / N
            p_val = 1.0 - poisson.cdf(shared - 1, exp_shared) if exp_shared > 0 else 1.0

            if p_val <= max_p_pair and sim >= min_sim:
                G.add_edge(s1 + 1, s2 + 1, weight=sim, p_value=p_val, shared=shared)
                edge_list.append({
                    "site_u": s1 + 1,
                    "site_v": s2 + 1,
                    "ref_u": REV_AA_MAP.get(consensus[s1], '-'),
                    "ref_v": REV_AA_MAP.get(consensus[s2], '-'),
                    "similarity": sim,
                    "p_value": float(p_val),
                    "shared_taxa": shared
                })

    # TSE Stage 2: Maximal Clique Seed Discovery
    raw_cliques = list(nx.find_cliques(G))
    cliques = [c for c in raw_cliques if len(c) >= min_clique_size]
    cliques.sort(key=lambda c: len(c), reverse=True)

    sectors = []
    seen_subsets = set()

    for c_idx, clq in enumerate(cliques[:max_sectors]):
        clq_sorted = sorted(clq)
        clq_key = tuple(clq_sorted)
        if clq_key in seen_subsets:
            continue
        seen_subsets.add(clq_key)

        site_0based = [s - 1 for s in clq_sorted]
        sub_B = B[site_0based, :]
        cov = sub_B @ sub_B.T
        eigvals = np.linalg.eigvalsh(cov)
        coherence = float(eigvals[-1] / (np.sum(eigvals) + 1e-15))

        # Multi-way joint support
        u = np.min(sub_B, axis=0)
        simul_taxa = int(np.sum(u > 0))
        exp_simul = N * np.prod([np.sum(B[s, :] > 0) / N for s in site_0based])
        p_multi = float(1.0 - poisson.cdf(simul_taxa - 1, exp_simul) if exp_simul > 0 else 1.0)

        # PARS residue signature
        pars_elements = []
        for s in site_0based:
            ref = REV_AA_MAP.get(consensus[s], '-')
            mut_taxa = np.where(B[s, :] > 0)[0]
            mut_aas = [REV_AA_MAP.get(aa_mat[i, s], '-') for i in mut_taxa if aa_mat[i, s] < 20]
            derived = max(set(mut_aas), key=mut_aas.count) if mut_aas else ref
            pars_elements.append(f"{ref}{s+1}{derived}")

        compact_pars = f"[ {' - '.join(pars_elements)} ]"

        sectors.append({
            "sector_id": len(sectors) + 1,
            "size": len(clq_sorted),
            "sites": clq_sorted,
            "coherence": coherence,
            "simultaneous_taxa": simul_taxa,
            "expected_simultaneous": float(exp_simul),
            "multi_way_p_value": p_multi,
            "pars_signature": compact_pars
        })

    return {
        "alignment": alignment_path,
        "taxa_count": N,
        "codon_count": L,
        "active_sites_count": len(active_sites),
        "coevolution_edges_count": len(edge_list),
        "sectors_discovered": len(sectors),
        "sectors": sectors,
        "edges": edge_list
    }
