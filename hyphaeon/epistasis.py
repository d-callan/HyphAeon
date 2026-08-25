"""
axomeme/epistasis.py
--------------------
Phylogenetic Branch Attribution, Inter-Site Co-Selection Networks, 
In Silico Selection Deep Mutational Scanning (Selection DMS / ESSM),
and Multi-Scale Epistatic Sector Mining.
"""

import os
import sys
import json
import math
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import networkx as nx
from Bio import Phylo

from .dataset import (
    GENETIC_CODE,
    AA_MAP,
    CODON_TO_AA,
    load_alignment_and_tree,
    get_codon_token,
    get_aa_token
)
from .model import PhyloAxialTransformer
from .weights import load_weights, load_arch_config

REV_AA_MAP = {v: k for k, v in AA_MAP.items()}

# Canonical sense codons for all 20 standard amino acids
CANONICAL_AA_TO_CODON = {
    'A': 'GCC', 'C': 'TGC', 'D': 'GAC', 'E': 'GAG', 'F': 'TTC',
    'G': 'GGC', 'H': 'CAC', 'I': 'ATC', 'K': 'AAG', 'L': 'CTG',
    'M': 'ATG', 'N': 'AAC', 'P': 'CCC', 'Q': 'CAG', 'R': 'CGC',
    'S': 'AGC', 'T': 'ACC', 'V': 'GTG', 'W': 'TGG', 'Y': 'TAC'
}

def build_clade_membership_matrix(tree_obj: Phylo.BaseTree.Tree, taxa: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Traverses all directed branches (clades) in the phylogenetic tree and constructs
    the binary clade membership indicator matrix C [num_branches, num_taxa].
    Each row C[k, :] = 1 for all terminal taxa subtended by branch k.
    """
    taxa_to_idx = {name: idx for idx, name in enumerate(taxa)}
    n_taxa = len(taxa)
    
    # Get all non-root clades (each corresponding to a directed branch in the tree)
    branches = []
    clade_rows = []
    
    for idx, clade in enumerate(tree_obj.find_clades(order='preorder')):
        if clade == tree_obj.root:
            continue
        terminals = [term.name.strip("'\"") for term in clade.get_terminals() if term.name]
        matching_indices = [taxa_to_idx[t] for t in terminals if t in taxa_to_idx]
        
        # Only retain branches that subtend at least 1 and at most N-1 taxa
        if 1 <= len(matching_indices) < n_taxa:
            row = np.zeros(n_taxa, dtype=np.float32)
            row[matching_indices] = 1.0
            branch_label = clade.name if clade.name else f"Branch_{len(branches) + 1}"
            branches.append(branch_label)
            clade_rows.append(row)
            
    if not clade_rows:
        # Fallback to identity matrix (terminal tip branches)
        C = np.eye(n_taxa, dtype=np.float32)
        branches = [f"Tip_{t}" for t in taxa]
    else:
        C = np.array(clade_rows, dtype=np.float32)
        
    return C, branches

def compute_phylogenetic_branch_attributions(
    model: PhyloAxialTransformer,
    c_tensor: torch.Tensor,
    a_tensor: torch.Tensor,
    tree_cache: Dict[str, Any],
    tree_obj: Phylo.BaseTree.Tree,
    taxa: List[str],
    device: torch.device,
    batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Evaluates baseline selection LRTs, extracts multi-head root-to-leaf attention weights,
    and projects leaf mutational evidence onto phylogenetic branches.
    Returns (branch_attributions [L, M], leaf_attributions [L, N], lrts [L], pvals [L], consensus_aas [L], branch_names [M]).
    """
    L, n_taxa, _ = c_tensor.shape
    C, branch_names = build_clade_membership_matrix(tree_obj, taxa)
    M = C.shape[0]
    
    # 1. Determine consensus amino acid per site
    a_np = a_tensor.squeeze(-1).numpy() # [L, N]
    consensus_aas = []
    for s in range(L):
        valid = a_np[s][a_np[s] < 20]
        if len(valid) > 0:
            major_aa = int(np.argmax(np.bincount(valid)))
            consensus_aas.append(REV_AA_MAP.get(major_aa, '-'))
        else:
            consensus_aas.append('-')
            
    # 2. Mutational indicator matrix delta [L, N]
    delta = np.zeros((L, n_taxa), dtype=np.float32)
    for s in range(L):
        cons_tok = AA_MAP.get(consensus_aas[s], 20)
        if cons_tok < 20:
            for n in range(n_taxa):
                aa_val = a_np[s, n]
                if aa_val < 20 and aa_val != cons_tok:
                    delta[s, n] = 1.0
                    
    # 3. Batched neural inference with root attention extraction
    lrts = np.zeros(L, dtype=np.float32)
    mean_attns = np.zeros((L, n_taxa), dtype=np.float32)
    
    model.eval()
    with torch.no_grad():
        for start_idx in range(0, L, batch_size):
            end_idx = min(start_idx + batch_size, L)
            c_chunk = c_tensor[start_idx:end_idx].to(device)
            a_chunk = a_tensor[start_idx:end_idx].to(device)
            
            y_soft, _, root_attns = model.forward_cached(
                c_chunk, a_chunk, tree_cache, return_attentions=True
            )
            lrts[start_idx:end_idx] = torch.clamp(y_soft, min=0.0).cpu().numpy().flatten()
            mean_attns[start_idx:end_idx] = root_attns.cpu().numpy()
            
    # 4. Asymptotic p-values
    pvals = np.ones(L, dtype=np.float32)
    pos_mask = lrts > 0.0
    pvals[pos_mask] = 0.5 * stats.chi2.sf(lrts[pos_mask], df=1)
    
    # 5. Leaf attributions: a_{n, i} = alpha_{n, i} * delta_{n, i}
    leaf_attributions = mean_attns * delta # [L, N]
    
    # 6. Branch projections: B = A @ C.T [L, M]
    branch_attributions = leaf_attributions.dot(C.T) # [L, M]
    
    return branch_attributions, leaf_attributions, lrts, pvals, consensus_aas, branch_names

def compute_branch_coselection_network(
    branch_attributions: np.ndarray,
    lrts: np.ndarray,
    branch_names: List[str],
    consensus_aas: List[str],
    min_sim: float = 0.30,
    min_shared: int = 2,
    max_fdr: float = 0.05,
    min_lrt: float = 1.0
) -> Tuple[List[Dict[str, Any]], nx.Graph]:
    """
    Computes pairwise phylogenetic branch co-selection cosine similarity,
    exact tree hypergeometric p-values, Benjamini-Hochberg FDR q-values,
    and Composite Epistatic Selection Index (CESI).
    """
    L, M = branch_attributions.shape
    
    # Identify active polymorphic sites with at least 1 mutated branch
    active_sites = [s for s in range(L) if np.sum(branch_attributions[s, :] > 0) >= 1]
    
    candidate_pairs = []
    G = nx.Graph()
    
    for s in range(L):
        G.add_node(s + 1, ref=consensus_aas[s], lrt=float(lrts[s]))
        
    # Vectorized computation of pairwise dot products and shared branch intersections
    norms = np.linalg.norm(branch_attributions, axis=1)  # [L]
    k_counts = np.sum(branch_attributions > 0, axis=1)   # [L]
    
    active_mask = (norms > 0)
    valid_sites = np.where(active_mask)[0]
    if len(valid_sites) < 2:
        return [], G
        
    B_active = branch_attributions[valid_sites]  # [V, M]
    B_bin = (B_active > 0).astype(np.float32)    # [V, M]
    norms_active = norms[valid_sites]            # [V]
    k_active = k_counts[valid_sites]             # [V]
    
    dot_matrix = B_active @ B_active.T           # [V, V]
    shared_matrix = B_bin @ B_bin.T              # [V, V]
    norm_outer = np.outer(norms_active, norms_active)
    sim_matrix = dot_matrix / np.maximum(norm_outer, 1e-9)
    
    tri_i, tri_j = np.triu_indices(len(valid_sites), k=1)
    shared_arr = shared_matrix[tri_i, tri_j]
    sim_arr = sim_matrix[tri_i, tri_j]
    s1_arr = valid_sites[tri_i]
    s2_arr = valid_sites[tri_j]
    
    lrt_s1 = lrts[s1_arr]
    lrt_s2 = lrts[s2_arr]
    max_lrt_arr = np.maximum(lrt_s1, lrt_s2)
    
    pass_filter = (shared_arr >= min_shared) & (sim_arr >= min_sim) & (max_lrt_arr >= min_lrt)
    pass_indices = np.where(pass_filter)[0]
    
    for idx in pass_indices:
        s1 = int(s1_arr[idx])
        s2 = int(s2_arr[idx])
        shared = int(shared_arr[idx])
        sim = float(sim_arr[idx])
        k1 = int(k_active[tri_i[idx]])
        k2 = int(k_active[tri_j[idx]])
        
        p_hyper = float(stats.hypergeom.sf(shared - 1, M, k1, k2))
        cesi = sim * math.sqrt(max(float(lrts[s1]), 0.1) * max(float(lrts[s2]), 0.1)) * math.log10(1.0 + shared)
        
        candidate_pairs.append({
            "site_u": s1 + 1,
            "site_v": s2 + 1,
            "ref_u": consensus_aas[s1],
            "ref_v": consensus_aas[s2],
            "lrt_u": float(lrts[s1]),
            "lrt_v": float(lrts[s2]),
            "similarity": sim,
            "shared_branches": shared,
            "branches_u": k1,
            "branches_v": k2,
            "cesi": float(cesi),
            "p_hyper": p_hyper
        })
        
    if not candidate_pairs:
        return [], G
        
    # Multiple testing FDR correction across evaluated candidate pairs
    p_vals = np.array([p["p_hyper"] for p in candidate_pairs], dtype=np.float32)
    n_pairs = len(p_vals)
    order = np.argsort(p_vals)
    ranks = np.empty(n_pairs, dtype=int)
    ranks[order] = np.arange(1, n_pairs + 1)
    raw_q = p_vals * (n_pairs / ranks)
    sorted_q = raw_q[order]
    for i in range(n_pairs - 2, -1, -1):
        sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
    raw_q[order] = sorted_q
    q_vals = np.clip(raw_q, 0.0, 1.0)
    
    reportable_edges = []
    for idx, p in enumerate(candidate_pairs):
        q = float(q_vals[idx])
        p["fdr_q"] = q
        if q <= max_fdr:
            reportable_edges.append(p)
            G.add_edge(
                p["site_u"], p["site_v"],
                weight=p["similarity"],
                cesi=p["cesi"],
                shared=p["shared_branches"],
                fdr_q=q
            )
            
    # Sort by CESI descending
    reportable_edges.sort(key=lambda e: e["cesi"], reverse=True)
    return reportable_edges, G

def compute_selection_dms_essm(
    model: PhyloAxialTransformer,
    c_tensor: torch.Tensor,
    a_tensor: torch.Tensor,
    tree_cache: Dict[str, Any],
    taxa: List[str],
    focal_taxon_idx: int = 0,
    target_sites: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 64
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Executes in silico Selection Deep Mutational Scanning (Selection DMS).
    Sweeps all 19 alternative amino acids at designated target sites in the focal taxon,
    measures differential selection shifts Delta LRT, and constructs the
    Epistatic Selection Sensitivity Matrix (ESSM).
    """
    if device is None:
        device = torch.device('cpu')
        
    L, n_taxa, _ = c_tensor.shape
    if target_sites is None:
        target_sites = list(range(L))
        
    # 1. Evaluate baseline LRT profile
    model.eval()
    baseline_lrts = np.zeros(L, dtype=np.float32)
    with torch.no_grad():
        for start_idx in range(0, L, batch_size):
            end_idx = min(start_idx + batch_size, L)
            c_chunk = c_tensor[start_idx:end_idx].to(device)
            a_chunk = a_tensor[start_idx:end_idx].to(device)
            y_soft, _ = model.forward_cached(c_chunk, a_chunk, tree_cache)
            baseline_lrts[start_idx:end_idx] = torch.clamp(y_soft.squeeze(-1), min=0.0).cpu().numpy().flatten()
            
    # 2. Epistatic Sensitivity Matrix E [len(target_sites), L]
    n_targets = len(target_sites)
    E_matrix = np.zeros((n_targets, L), dtype=np.float32)
    plasticity_records = []
    
    standard_aas = list(CANONICAL_AA_TO_CODON.keys())
    
    # Batch site perturbations in chunks for maximum GPU acceleration
    chunk_sites_num = max(1, batch_size // 19)
    with torch.no_grad():
        for chunk_start in range(0, n_targets, chunk_sites_num):
            chunk_target_sites = target_sites[chunk_start:chunk_start + chunk_sites_num]
            
            c_batch_list = []
            a_batch_list = []
            meta_list = []
            
            for site in chunk_target_sites:
                orig_c = int(c_tensor[site, focal_taxon_idx, 0])
                orig_a = int(a_tensor[site, focal_taxon_idx, 0])
                wt_aa = REV_AA_MAP.get(orig_a, 'A')
                alt_aas = [aa for aa in standard_aas if aa != wt_aa]
                
                c_site = c_tensor[site:site+1].repeat(len(alt_aas), 1, 1).clone()
                a_site = a_tensor[site:site+1].repeat(len(alt_aas), 1, 1).clone()
                
                for m_idx, alt_aa in enumerate(alt_aas):
                    alt_codon = CANONICAL_AA_TO_CODON[alt_aa]
                    c_site[m_idx, focal_taxon_idx, 0] = get_codon_token(alt_codon)
                    a_site[m_idx, focal_taxon_idx, 0] = get_aa_token(alt_codon)
                    
                c_batch_list.append(c_site)
                a_batch_list.append(a_site)
                meta_list.append((site, wt_aa, len(alt_aas)))
                
            c_batch = torch.cat(c_batch_list, dim=0).to(device)
            a_batch = torch.cat(a_batch_list, dim=0).to(device)
            
            y_soft_pert, _ = model.forward_cached(c_batch, a_batch, tree_cache)
            all_pert_lrts = torch.clamp(y_soft_pert.squeeze(-1), min=0.0).cpu().numpy().flatten()
            
            offset = 0
            for s_idx, (site, wt_aa, n_alts) in enumerate(meta_list):
                pert_lrts = all_pert_lrts[offset:offset + n_alts]
                offset += n_alts
                
                self_shifts = np.abs(pert_lrts - baseline_lrts[site])
                intrinsic_plasticity = float(np.mean(self_shifts))
                max_self_shift = float(np.max(self_shifts))
                
                t_idx = chunk_start + s_idx
                E_matrix[t_idx, site] = intrinsic_plasticity
                
                p_val = float(0.5 * stats.chi2.sf(baseline_lrts[site], df=1)) if baseline_lrts[site] > 0 else 1.0
                
                plasticity_records.append({
                    "site": site + 1,
                    "wt_aa": wt_aa,
                    "focal_taxon": taxa[focal_taxon_idx],
                    "baseline_lrt": float(baseline_lrts[site]),
                    "p_value": p_val,
                    "intrinsic_plasticity": intrinsic_plasticity,
                    "max_shift": max_self_shift
                })
            
    df_plasticity = pd.DataFrame(plasticity_records)
    return df_plasticity, E_matrix

def extract_epistatic_sectors(
    G: nx.Graph,
    branch_attributions: np.ndarray,
    lrts: np.ndarray,
    consensus_aas: List[str],
    min_clique_size: int = 3,
    max_sectors: int = 15,
    max_overlap: float = 0.50
) -> List[Dict[str, Any]]:
    """
    Extracts distinct, non-redundant epistatic sectors from the co-selection graph
    using maximal clique decomposition, Jaccard overlap suppression, and spectral coherence.
    """
    import itertools
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return []
        
    # Decompose dense graphs into modular communities to avoid exponential Bron-Kerbosch explosion
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
    except Exception:
        communities = [set(c) for c in nx.connected_components(G)]
        
    candidate_cliques = []
    for comm in communities:
        if len(comm) < min_clique_size:
            continue
        subG = G.subgraph(comm)
        for clq in itertools.islice(nx.find_cliques(subG), 500):
            if len(clq) >= min_clique_size:
                candidate_cliques.append(clq)
                
    if not candidate_cliques:
        for clq in itertools.islice(nx.find_cliques(G), 1000):
            if len(clq) >= min_clique_size:
                candidate_cliques.append(clq)
                
    candidate_cliques.sort(key=lambda c: len(c), reverse=True)
    
    sectors = []
    
    for clq in candidate_cliques:
        if len(sectors) >= max_sectors:
            break
        clq_sorted = sorted(clq)
        clq_set = set(clq_sorted)
        
        # Jaccard Overlap Suppression: filter redundant permutations of already discovered sectors
        is_redundant = False
        for sec in sectors:
            sec_set = set(sec["sites"])
            jaccard = len(clq_set & sec_set) / max(1, len(clq_set | sec_set))
            if jaccard > max_overlap:
                is_redundant = True
                break
        if is_redundant:
            continue
        
        site_0based = [s - 1 for s in clq_sorted]
        sub_B = branch_attributions[site_0based, :] # [k, M]
        
        # Spectral Coherence: lambda_1 / sum(lambda) of correlation matrix
        cov = sub_B.dot(sub_B.T)
        eigvals = np.linalg.eigvalsh(cov)
        coherence = float(eigvals[-1] / (np.sum(eigvals) + 1e-15)) if len(eigvals) > 0 else 1.0
        
        # Multi-way shared branch support
        shared_mask = np.min(sub_B > 0, axis=0)
        simul_branches = int(np.sum(shared_mask))
        
        # PARS residue signature: [ WT_site_Derived ]
        pars_tokens = []
        for s in site_0based:
            ref = consensus_aas[s]
            pars_tokens.append(f"{ref}{s+1}")
        signature = f"[ {' - '.join(pars_tokens)} ]"
        
        mean_lrt = float(np.mean(lrts[site_0based]))
        
        sectors.append({
            "sector_id": len(sectors) + 1,
            "size": len(clq_sorted),
            "sites": clq_sorted,
            "mean_lrt": mean_lrt,
            "spectral_coherence": coherence,
            "shared_branches": simul_branches,
            "pars_signature": signature
        })
        
    return sectors

def run_epistasis_analysis(
    alignment_path: str,
    tree_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    focal_taxon: Optional[str] = None,
    min_sim: float = 0.30,
    min_shared: int = 2,
    max_fdr: float = 0.05,
    min_lrt: float = 1.0,
    min_clique_size: int = 3,
    max_overlap: float = 0.50,
    run_dms: bool = True,
    cpu: bool = False
) -> Dict[str, Any]:
    """
    Master pipeline executing Phylogenetic Branch Attribution, Inter-Site Co-Selection,
    Selection DMS (ESSM), and Epistatic Sector Mining.
    """
    if torch.cuda.is_available() and not cpu:
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and not cpu:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    # 1. Load Alignment, Tree, and Extract Tree Cache
    c, a, d, z, inv, taxa, L = load_alignment_and_tree(alignment_path, tree_path, prune_duplicates=True)
    tree_obj = Phylo.read(tree_path, 'newick') if tree_path and os.path.exists(tree_path) else None
    if tree_obj is None:
        # Re-parse tree from file
        from .dataset import extract_tree_from_string_or_file
        tree_obj = extract_tree_from_string_or_file(tree_path if tree_path else alignment_path)
        
    # 2. Load Model
    config = load_arch_config(weights=weights_path)
    model = PhyloAxialTransformer(
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        window_size=config['window_size'],
    ).to(device)
    state_dict = load_weights(weights=weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    d_dev = d.to(device)
    z_dev = z.to(device)
    tree_cache = model.precompute_tree_cache(d_dev, z_dev)
    
    # 3. Compute Phylogenetic Branch Attributions
    branch_attr, leaf_attr, lrts, pvals, cons_aas, branch_names = compute_phylogenetic_branch_attributions(
        model, c, a, tree_cache, tree_obj, taxa, device
    )
    
    # 4. Compute Inter-Site Co-Selection Network
    edges, G = compute_branch_coselection_network(
        branch_attr, lrts, branch_names, cons_aas,
        min_sim=min_sim, min_shared=min_shared, max_fdr=max_fdr, min_lrt=min_lrt
    )
    
    # 5. Extract Epistatic Sectors
    sectors = extract_epistatic_sectors(
        G, branch_attr, lrts, cons_aas, min_clique_size=min_clique_size, max_overlap=max_overlap
    )
    
    # 6. Optional In Silico Selection DMS (ESSM)
    df_plasticity = None
    if run_dms:
        focal_idx = taxa.index(focal_taxon) if focal_taxon and focal_taxon in taxa else 0
        df_plasticity, _ = compute_selection_dms_essm(
            model, c, a, tree_cache, taxa, focal_taxon_idx=focal_idx, device=device
        )
        
    return {
        "alignment": alignment_path,
        "taxa_count": len(taxa),
        "codon_count": L,
        "branch_count": len(branch_names),
        "coevolution_edges_count": len(edges),
        "sectors_discovered": len(sectors),
        "edges": edges,
        "sectors": sectors,
        "plasticity": df_plasticity.to_dict(orient="records") if df_plasticity is not None else []
    }

# Alias for backward compatibility
run_epistatic_sector_mining = run_epistasis_analysis
