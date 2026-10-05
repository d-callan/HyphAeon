"""
hyphaeon/splits.py
------------------
Spectral Graph Bisection CLI logic (HyphAeon-specific).

Cross-taxa attention extraction and fused affinity matrix computation
now live in aeon_core.splits.
"""

import os
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch

from aeon_core.inference import load_model, get_device
from aeon_core.dataset import load_alignment_and_tree
from aeon_core.splits import (
    extract_cross_taxa_attentions_and_embeddings,
    compute_fused_affinity_matrix,
)


def spectral_bisection(
    affinity_matrix: np.ndarray,
    taxa_names: List[str],
    min_clade_size: int = 2,
    max_depth: int = 10,
    current_depth: int = 0
) -> Dict[str, Any]:
    """
    Recursively partitions taxa using the Fiedler vector of the normalized graph Laplacian.

    Args:
        affinity_matrix: (N x N) symmetric non-negative affinity matrix.
        taxa_names: List of N taxon identifiers.
        min_clade_size: Minimum number of taxa in a clade before terminating bisection.
        max_depth: Maximum recursion depth.

    Returns:
        Hierarchical tree dictionary with node attributes:
          - fiedler_val: 2nd smallest eigenvalue (algebraic connectivity).
          - eigengap: lambda_3 - lambda_2 (split stability margin).
          - cut_weight: sum of inter-clade edge weights.
          - left, right: child clade nodes.
    """
    n = len(taxa_names)
    if n <= min_clade_size or current_depth >= max_depth:
        return {"type": "leaf", "taxa": taxa_names}

    A = (affinity_matrix + affinity_matrix.T) / 2.0
    np.fill_diagonal(A, 0.0)

    d = A.sum(axis=1)
    d[d == 0] = 1e-8
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # L_sym = I - D^(-1/2) A D^(-1/2)
    L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    evals, evecs = np.linalg.eigh(L_sym)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    fiedler_val = float(evals[1]) if n > 1 else 0.0
    fiedler_vec = evecs[:, 1] if n > 1 else np.zeros(n)
    eigengap = float(evals[2] - evals[1]) if n > 2 else float(evals[1]) if n > 1 else 0.0

    # Unnormalized indicator y = D^(-1/2) v_2
    y = d_inv_sqrt * fiedler_vec

    left_mask = (y >= 0)
    right_mask = ~left_mask

    if left_mask.sum() == 0 or right_mask.sum() == 0:
        median_val = np.median(y)
        left_mask = (y >= median_val)
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            left_mask[:n // 2] = True
            right_mask = ~left_mask

    left_taxa = [taxa_names[i] for i in range(n) if left_mask[i]]
    right_taxa = [taxa_names[i] for i in range(n) if right_mask[i]]

    cut_weight = float(A[left_mask][:, right_mask].sum())

    A_left = A[np.ix_(left_mask, left_mask)]
    A_right = A[np.ix_(right_mask, right_mask)]

    left_child = spectral_bisection(A_left, left_taxa, min_clade_size, max_depth, current_depth + 1)
    right_child = spectral_bisection(A_right, right_taxa, min_clade_size, max_depth, current_depth + 1)

    return {
        "type": "node",
        "fiedler_val": fiedler_val,
        "eigengap": eigengap,
        "cut_weight": cut_weight,
        "depth": current_depth,
        "taxa_count": n,
        "left": left_child,
        "right": right_child
    }


def tree_dict_to_newick(tree_dict: Dict[str, Any]) -> str:
    """Converts a hierarchical bisection tree dictionary into a formatted Newick string."""
    if tree_dict["type"] == "leaf":
        if len(tree_dict["taxa"]) == 1:
            return tree_dict["taxa"][0]
        else:
            return "(" + ",".join(tree_dict["taxa"]) + ")"
    left_nwk = tree_dict_to_newick(tree_dict["left"])
    right_nwk = tree_dict_to_newick(tree_dict["right"])
    support = tree_dict.get("eigengap", 0.0)
    return f"({left_nwk},{right_nwk}):{support:.4f}"


def get_all_clade_taxa(node: Dict[str, Any]) -> List[str]:
    """Helper to collect all leaf taxa beneath a tree node."""
    if node["type"] == "leaf":
        return node["taxa"]
    return get_all_clade_taxa(node["left"]) + get_all_clade_taxa(node["right"])


def run_spectral_splits(
    alignment_path: str,
    tree_path: Optional[str] = None,
    use_tn93: bool = False,
    weights_path: Optional[str] = None,
    min_clade_size: int = 2,
    max_depth: int = 10,
    device: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    End-to-end pipeline to recover well-supported phylogenetic splits via spectral bisection.

    Args:
        alignment_path: Path to FASTA alignment.
        tree_path: Optional path to Newick tree.
        use_tn93: If True, computes pairwise TN93 distance matrix directly (skips tree).
        weights_path: Path to local weights file; if None, resolves the HF variant.
        min_clade_size: Clade size floor.
        max_depth: Max tree depth.
        device: PyTorch device.

    Returns:
        Dictionary with derived Newick string, root split clades, Fiedler value, eigengap, and tree dict.
    """
    if device is None:
        device = get_device(cpu=True)

    model = load_model(weights=weights_path, device=device)

    c_tensor, a_tensor, d_tensor, z_tensor, _, taxa, L = load_alignment_and_tree(
        alignment_path, nwk_path=tree_path, use_tn93=use_tn93, prune_duplicates=False
    )

    msa_codons = c_tensor.to(device)
    msa_aas = a_tensor.to(device)
    dist_mat = d_tensor.squeeze(0).cpu().numpy()
    mds_coords = z_tensor.squeeze(0).cpu().numpy()

    tree_cache = model.precompute_tree_cache(dist_mat, mds_coords)

    cross_attn, taxa_repr = extract_cross_taxa_attentions_and_embeddings(
        model, msa_codons, msa_aas, tree_cache, device=device
    )

    A_fused = compute_fused_affinity_matrix(cross_attn, mds_coords, taxa_repr)
    split_tree = spectral_bisection(A_fused, taxa, min_clade_size=min_clade_size, max_depth=max_depth)
    newick_str = tree_dict_to_newick(split_tree) + ";"

    left_taxa = get_all_clade_taxa(split_tree["left"])
    right_taxa = get_all_clade_taxa(split_tree["right"])

    return {
        "newick": newick_str,
        "fiedler_val": split_tree.get("fiedler_val", 0.0),
        "eigengap": split_tree.get("eigengap", 0.0),
        "cut_weight": split_tree.get("cut_weight", 0.0),
        "root_split": {
            "left_clade": left_taxa,
            "right_clade": right_taxa,
            "left_count": len(left_taxa),
            "right_count": len(right_taxa)
        },
        "taxa": taxa,
        "L": L,
        "tree_dict": split_tree
    }
