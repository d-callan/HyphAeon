"""
axomeme/epistasis.py
--------------------
Phylogenetic Branch Attribution, Epistatic Co-Selection Networks,
and In Silico Selection Deep Mutational Scanning (ESSM / Digital DMS).

Uses exact Fitch parsimony ancestral state reconstruction combined with
root-to-leaf multi-head axial attention weights to map specific mutational
events to tree branches and discover epistatic co-evolutionary sectors.
"""

import os
import sys
import math
import json
import time
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import networkx as nx
from Bio import Phylo

from .dataset import (
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

def fitch_branch_substitution_mapping(
    tree_obj: Phylo.BaseTree.Tree,
    a_np: np.ndarray,
    taxa_to_idx: Dict[str, int],
    attns: np.ndarray,
    L: int
) -> Tuple[np.ndarray, List[Phylo.BaseTree.Clade], List[str]]:
    """
    Performs Fitch maximum parsimony ancestral state reconstruction across the tree
    to map amino acid substitutions to specific directed phylogenetic branches.
    Returns:
      branch_attributions: [L, M] matrix of attention-weighted branch substitutions
      branches: list of M child clades corresponding to directed branches
      branch_names: list of M branch descriptor strings
    """
    # Build parent map and branch list
    parent_map = {}
    branches = []
    clade_to_idx = {}
    for clade in tree_obj.find_clades(order='preorder'):
        for child in clade.clades:
            parent_map[child] = clade
            branches.append(child)
            clade_to_idx[child] = len(branches) - 1

    M = len(branches)
    branch_names = [b.name if b.name else f"Branch_{i+1}" for i, b in enumerate(branches)]
    B_branch = np.zeros((L, M), dtype=np.float32)

    for s in range(L):
        # 1. Map leaf states
        leaf_states = {}
        for clade in tree_obj.get_terminals():
            t_name = clade.name.strip("'\"")
            if t_name in taxa_to_idx:
                idx = taxa_to_idx[t_name]
                tok = a_np[s, idx]
                leaf_states[clade] = {tok} if tok < 20 else set()
            else:
                leaf_states[clade] = set()

        # 2. Bottom-up post-order traversal (candidate state sets)
        fitch_sets = {}
        for clade in tree_obj.find_clades(order='postorder'):
            if clade.is_terminal():
                fitch_sets[clade] = leaf_states.get(clade, set())
            else:
                child_sets = [fitch_sets[c] for c in clade.clades if len(fitch_sets[c]) > 0]
                if not child_sets:
                    fitch_sets[clade] = set()
                else:
                    inter = set.intersection(*child_sets)
                    fitch_sets[clade] = inter if inter else set.union(*child_sets)

        # 3. Top-down pre-order traversal (assign parsimonious states & detect substitutions)
        assigned = {}
        root_s = fitch_sets[tree_obj.root]
        assigned[tree_obj.root] = next(iter(root_s)) if root_s else 20

        for clade in tree_obj.find_clades(order='preorder'):
            if clade == tree_obj.root:
                continue
            p_state = assigned[parent_map[clade]]
            c_set = fitch_sets[clade]
            if p_state in c_set:
                assigned[clade] = p_state
            elif c_set:
                assigned[clade] = next(iter(c_set))
            else:
                assigned[clade] = p_state

            # Substitution event occurs on this branch when child state != parent state
            if assigned[clade] != p_state and assigned[clade] < 20 and p_state < 20:
                b_idx = clade_to_idx[clade]
                desc_leaves = [taxa_to_idx[t.name.strip('\'\"')] for t in clade.get_terminals() if t.name.strip('\'\"') in taxa_to_idx]
                attn_weight = float(np.mean(attns[s, desc_leaves])) if (desc_leaves and attns is not None) else 1.0
                B_branch[s, b_idx] = max(1e-4, attn_weight)

    return B_branch, branches, branch_names

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
    and reconstructs exact branch-level substitution attributions using Fitch parsimony.
    Returns (branch_attributions [L, M], leaf_attributions [L, N], lrts [L], pvals [L], consensus_aas [L], branch_names [M]).
    """
    L, n_taxa, _ = c_tensor.shape
    taxa_to_idx = {name: idx for idx, name in enumerate(taxa)}
    
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
    
    # 6. Branch substitutions: exact Fitch Parsimony on tree
    branch_attributions, _, branch_names = fitch_branch_substitution_mapping(
        tree_obj, a_np, taxa_to_idx, mean_attns, L
    )
    
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
    
    G = nx.Graph()
    for s in range(L):
        G.add_node(s + 1, ref=consensus_aas[s], lrt=float(lrts[s]))
        
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
    
    candidate_pairs = []
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
            "hyper_p": p_hyper,
            "cesi": float(cesi)
        })
        
    if not candidate_pairs:
        return [], G
        
    # Sort candidate pairs by hypergeometric p-value for Benjamini-Hochberg FDR
    candidate_pairs.sort(key=lambda x: x["hyper_p"])
    m_tests = len(candidate_pairs)
    sig_pairs = []
    min_q = 1.0
    
    for rank, p in reversed(list(enumerate(candidate_pairs))):
        q = (p["hyper_p"] * m_tests) / (rank + 1)
        if q < min_q:
            min_q = q
        p["fdr_q"] = min(min_q, 1.0)
        
    for p in candidate_pairs:
        if p["fdr_q"] <= max_fdr:
            sig_pairs.append(p)
            G.add_edge(
                p["site_u"],
                p["site_v"],
                weight=p["similarity"],
                shared=p["shared_branches"],
                cesi=p["cesi"],
                fdr_q=p["fdr_q"]
            )
            
    sig_pairs.sort(key=lambda x: x["cesi"], reverse=True)
    return sig_pairs, G

def extract_epistatic_sectors_tse(
    G: nx.Graph,
    branch_attributions: np.ndarray,
    lrts: np.ndarray,
    consensus_aas: List[str],
    min_clique_size: int = 3,
    max_overlap: float = 0.50
) -> List[Dict[str, Any]]:
    """
    Two-Stage Seed-and-Extend (TSE) epistatic sector mining.
    Uses greedy modularity community decomposition on the significant co-selection graph
    to find dense, non-overlapping epistatic functional sectors.
    """
    if G.number_of_edges() == 0:
        return []
        
    # Remove isolated nodes
    sub_nodes = [n for n, d in G.degree() if d > 0]
    if len(sub_nodes) < min_clique_size:
        # Fallback to connected components
        components = [list(c) for c in nx.connected_components(G.subgraph(sub_nodes)) if len(c) >= 2]
    else:
        G_sub = G.subgraph(sub_nodes)
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(G_sub, weight='weight')
            components = [list(c) for c in communities if len(c) >= 2]
        except Exception:
            components = [list(c) for c in nx.connected_components(G_sub) if len(c) >= 2]
            
    sectors = []
    for comp_idx, members in enumerate(components):
        site_indices = [n - 1 for n in members]
        sub_B = branch_attributions[site_indices, :]
        
        # Spectral coherence C(S) = lambda_1 / sum(lambda_k)
        if sub_B.shape[0] >= 2 and np.linalg.norm(sub_B) > 0:
            cov = sub_B @ sub_B.T
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.maximum(eigvals, 0.0)
            tot_eig = np.sum(eigvals)
            coherence = float(eigvals[-1] / tot_eig) if tot_eig > 0 else 0.0
        else:
            coherence = 1.0
            
        shared_b = int(np.sum(np.all(sub_B > 0, axis=0)))
        mean_lrt_val = float(np.mean(lrts[site_indices]))
        
        sig_tokens = [f"{consensus_aas[s]}{s+1}" for s in sorted(site_indices)]
        pars_sig = f"[ {' - '.join(sig_tokens[:10])} ]"
        
        sectors.append({
            "sector_id": comp_idx + 1,
            "size": len(members),
            "sites": sorted(members),
            "spectral_coherence": coherence,
            "shared_branches": shared_b,
            "mean_lrt": mean_lrt_val,
            "pars_signature": pars_sig
        })
        
    sectors.sort(key=lambda x: (x["spectral_coherence"], x["size"]), reverse=True)
    return sectors

def run_insilico_selection_dms(
    model: PhyloAxialTransformer,
    c_tensor: torch.Tensor,
    a_tensor: torch.Tensor,
    tree_cache: Dict[str, Any],
    taxa: List[str],
    device: torch.device,
    focal_taxon: Optional[str] = None,
    batch_size: int = 64
) -> List[Dict[str, Any]]:
    """
    Performs 19-amino-acid in silico Deep Mutational Scanning (Selection DMS / ESSM)
    by computing delta LRT across all possible point substitutions at a focal taxon.
    """
    L, n_taxa, _ = c_tensor.shape
    
    # 1. Determine focal taxon index
    focal_idx = 0
    if focal_taxon:
        for idx, t in enumerate(taxa):
            if focal_taxon.lower() in t.lower():
                focal_idx = idx
                break
                
    # 2. Extract baseline LRTs and consensus WT residues
    a_np = a_tensor.squeeze(-1).numpy() # [L, N]
    model.eval()
    baseline_lrts = np.zeros(L, dtype=np.float32)
    
    with torch.no_grad():
        for start_idx in range(0, L, batch_size):
            end_idx = min(start_idx + batch_size, L)
            c_chunk = c_tensor[start_idx:end_idx].to(device)
            a_chunk = a_tensor[start_idx:end_idx].to(device)
            y_soft, _ = model.forward_cached(c_chunk, a_chunk, tree_cache)
            baseline_lrts[start_idx:end_idx] = torch.clamp(y_soft, min=0.0).cpu().numpy().flatten()
            
    # 3. Batched 19-AA mutational sweep
    plasticity_results = []
    
    standard_aas = list('ACDEFGHIKLMNPQRSTVWY')
    
    for s in range(L):
        wt_tok = a_np[s, focal_idx]
        wt_aa = REV_AA_MAP.get(wt_tok, '-')
        if wt_tok >= 20:
            valid = a_np[s][a_np[s] < 20]
            wt_tok = int(np.argmax(np.bincount(valid))) if len(valid) > 0 else 0
            wt_aa = REV_AA_MAP.get(wt_tok, 'A')
            
        cand_mut_aas = [aa for aa in standard_aas if aa != wt_aa]
        n_muts = len(cand_mut_aas)
        
        # Prepare batch of 19 mutated alignment tensors
        c_batch = c_tensor[s:s+1].repeat(n_muts, 1, 1).to(device)
        a_batch = a_tensor[s:s+1].repeat(n_muts, 1, 1).to(device)
        
        for m_i, m_aa in enumerate(cand_mut_aas):
            codon_str = CANONICAL_AA_TO_CODON[m_aa]
            c_tok = get_codon_token(codon_str)
            aa_tok = get_aa_token(codon_str)
            c_batch[m_i, focal_idx, 0] = c_tok
            a_batch[m_i, focal_idx, 0] = aa_tok
            
        with torch.no_grad():
            y_mut_soft, _ = model.forward_cached(c_batch, a_batch, tree_cache)
            mut_lrts = torch.clamp(y_mut_soft, min=0.0).cpu().numpy().flatten()
            
        delta_lrts = mut_lrts - baseline_lrts[s]
        plasticity = float(np.mean(np.abs(delta_lrts)))
        p_val = float(0.5 * stats.chi2.sf(max(0.0, baseline_lrts[s]), df=1))
        
        plasticity_results.append({
            "site": s + 1,
            "wt_aa": wt_aa,
            "baseline_lrt": float(baseline_lrts[s]),
            "p_value": p_val,
            "intrinsic_plasticity": plasticity,
            "mean_delta_lrt": float(np.mean(delta_lrts)),
            "max_delta_lrt": float(np.max(delta_lrts)),
            "min_delta_lrt": float(np.min(delta_lrts))
        })
        
    return plasticity_results

def run_epistatic_analysis(
    alignment_path: str,
    tree_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    variant: Optional[str] = None,
    focal_taxon: Optional[str] = None,
    min_sim: float = 0.30,
    min_shared: int = 2,
    max_fdr: float = 0.05,
    min_lrt: float = 1.0,
    min_clique_size: int = 3,
    max_overlap: float = 0.50,
    run_dms: bool = True,
    skip_dms: bool = False,
    cpu: bool = False,
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Executes complete Phylogenetic Branch Attribution, Co-Selection Networks,
    Epistatic Sector Mining, and Selection Deep Mutational Scanning (Digital DMS).
    """
    # 1. Device Selection
    if cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 2. Load Alignment and Tree
    c_tensor, a_tensor, d_mat, z_coords, inv_mask, taxa, L = load_alignment_and_tree(
        alignment_path, tree_path, prune_duplicates=True
    )
    tree_obj = Phylo.read(tree_path, 'newick') if tree_path and os.path.exists(tree_path) else None
    if tree_obj is None:
        from .dataset import extract_tree_from_string_or_file
        tree_obj = extract_tree_from_string_or_file(tree_path if tree_path else alignment_path)

    taxa_to_idx = {t: i for i, t in enumerate(taxa)}
    for clade in list(tree_obj.get_terminals()):
        if clade.name.strip('\'\"') not in taxa_to_idx:
            tree_obj.prune(clade)

    N = len(taxa)

    # 3. Load Model
    config = load_arch_config(weights=weights_path, variant=variant)
    model = PhyloAxialTransformer(
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        window_size=config['window_size']
    ).to(device)
    model.load_state_dict(load_weights(weights=weights_path, variant=variant, map_location=device), strict=False)
    model.eval()

    tree_cache = model.precompute_tree_cache(d_mat.to(device), z_coords.to(device))

    # 4. Compute Attributions and Co-Selection Network
    branch_attr, leaf_attr, lrts, pvals, consensus_aas, branch_names = compute_phylogenetic_branch_attributions(
        model, c_tensor, a_tensor, tree_cache, tree_obj, taxa, device, batch_size=batch_size
    )

    sig_edges, G = compute_branch_coselection_network(
        branch_attr, lrts, branch_names, consensus_aas,
        min_sim=min_sim, min_shared=min_shared, max_fdr=max_fdr, min_lrt=min_lrt
    )

    sectors = extract_epistatic_sectors_tse(
        G, branch_attr, lrts, consensus_aas,
        min_clique_size=min_clique_size, max_overlap=max_overlap
    )

    plasticity = []
    if run_dms and not skip_dms:
        plasticity = run_insilico_selection_dms(
            model, c_tensor, a_tensor, tree_cache, taxa, device,
            focal_taxon=focal_taxon, batch_size=batch_size
        )

    return {
        "alignment": alignment_path,
        "tree": tree_path,
        "taxa_count": N,
        "codon_count": L,
        "evaluated_branches": len(branch_names),
        "branch_count": len(branch_names),
        "coselection_edges_count": len(sig_edges),
        "discovered_sectors_count": len(sectors),
        "edges": sig_edges,
        "sectors": sectors,
        "plasticity": plasticity,
        "coselection_edges": sig_edges,
        "epistatic_sectors": sectors,
        "selection_dms_plasticity": plasticity
    }


# Backwards compatibility aliases
run_epistasis_analysis = run_epistatic_analysis
compute_selection_dms_essm = run_insilico_selection_dms
extract_epistatic_sectors = extract_epistatic_sectors_tse
run_epistatic_sector_mining = run_epistatic_analysis
