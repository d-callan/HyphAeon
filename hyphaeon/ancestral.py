"""
hyphaeon/ancestral.py
---------------------
EXPERIMENTAL / PROTOTYPE — HyphAeon-informed ancestral sequence candidate scoring.

This module does NOT attempt classical, tree-based ancestral sequence
reconstruction (ASR). HyphAeon is topology-blind (star-tree / permutation /
zero-distance invariant per model_eval/) and exposes no character-state decoder
head, so it cannot emit per-internal-node states the way PAML / HyPhy / IQ-TREE
marginal reconstruction can. See the design notes below.

Instead, this reframes the problem to match what the model actually is:

  * Because the model treats the taxa as a distance cloud (it barely uses
    topology), the natural object is a SINGLE "centroid ancestor" sitting at the
    origin of the model's 4D MDS distance embedding — exactly where the model
    itself places its internal [ROOT] token (dataset/model root_dist = ||z_n||).

  * The base reconstruction signal comes from the observed tips, weighted by
    patristic proximity to that centroid. This needs no model at all.

  * HyphAeon then enters ONLY as re-scorers / constraints on top of that base:
      - selection LRT  -> per-site reconstruction confidence
      - Digital DMS    -> per-state selection-signal sensitivity (a REGULARIZER)
      - co-selection   -> joint / epistatic compatibility across site pairs

The single most defensible contribution over standard ASR is the epistatic
term: classical marginal ASR reconstructs each site independently, whereas the
co-selection network lets us penalise ancestral candidates that combine
individually-plausible states that are jointly never observed / co-selected.

IMPORTANT CAVEATS (documented because the scoring is deliberately heuristic):
  1. DMS `delta_lrt` is a SELECTION-SIGNAL SENSITIVITY, not a fitness or a
     likelihood. It must not be used as "log P(state)". Here it is used only as
     a soft regulariser that discourages states which spike the diversifying
     selection signal (see `_dms_regulariser`). Do not turn it into the primary
     objective — optimising it directly would drive toward "no selection", which
     is meaningless as an ancestor.
  2. Without an outgroup the ancestor is an unrooted centroid (the MDS
     origin). Pass --outgroup to polarise: the ancestor is then placed at the
     outgroup's MDS position, so the outgroup naturally dominates the
     distance-weighted tip frequencies (distance 0 → highest weight).
  3. Amino-acid level only. DMS uses canonical codons, so reconstruction is at
     the amino-acid level; codon / synonymous reconstruction is out of scope.
  4. Clade bias. The DMS plasticity landscape reflects the model's training
     distribution (mammalian / viral). For divergent taxa the DMS and LRT terms
     may be miscalibrated — validate against a clade with a known ASR first.

The default weights below are heuristic prototype values, not tuned constants.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .dataset import (
    AA_MAP,
    CODON_TO_AA,
    GENETIC_CODE,
    load_alignment_and_tree,
)
from .epistasis import (
    CANONICAL_AA_TO_CODON,
    compute_branch_coselection_network,
    compute_transformer_attributions,
    run_insilico_selection_dms,
)
from .inference import get_device, load_model
from .dataset import compute_mds_coordinates

REV_AA_MAP: Dict[int, str] = {v: k for k, v in AA_MAP.items()}
GAP_TOKEN = 20  # amino-acid token used for gap / unknown in a_tensor
STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

# --- Prototype scoring weights (heuristic; not tuned) -----------------------
# base term always has weight 1.0; the following scale the model-derived terms
# relative to it. Kept small so the observed tips dominate and the model only
# breaks ties / enforces joint consistency.
DEFAULT_LAMBDA_DIST = 4.0      # steepness of centroid distance weighting exp(-lambda*||z||)
DEFAULT_LAMBDA_PLAST = 0.15    # weight of the DMS selection-sensitivity regulariser
DEFAULT_LAMBDA_EPI = 0.50      # weight of the epistatic co-occurrence term
DEFAULT_PSEUDOCOUNT = 1e-3     # additive smoothing for empirical frequencies


@dataclass
class AncestralCandidate:
    """One scored ancestral amino-acid sequence candidate."""
    sequence: str
    total_score: float
    base_score: float
    dms_score: float
    epi_score: float
    per_site_confidence: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "total_score": self.total_score,
            "base_score": self.base_score,
            "dms_score": self.dms_score,
            "epi_score": self.epi_score,
        }


# ---------------------------------------------------------------------------
# Stage 0: per-node anchor estimation (requires tree)
# ---------------------------------------------------------------------------
def compute_internal_node_anchors(
    tree_obj,
    taxa: List[str],
    z_coords: np.ndarray,
    outgroup_idx: Optional[int] = None,
) -> List[Tuple[str, np.ndarray, List[int]]]:
    """Estimate MDS anchor positions for each internal node of the tree.

    For each internal node, compute patristic distances from that node to all
    leaf taxa, then run classical MDS on those distances to get a 4D coordinate.
    The anchor is then snapped to the nearest existing tip MDS position if the
    node is a direct ancestor of only a subset of tips (clade-specific node).

    Returns a list of (node_label, anchor_z, descendant_tip_indices) tuples.
    When a tree is not available, returns a single entry for the root centroid.
    """
    from Bio import Phylo
    from io import StringIO

    # Build name -> tip clade lookup
    taxa_set = set(taxa)
    terminals = {}
    for t in tree_obj.get_terminals():
        if t.name and t.name.strip("'\"") in taxa_set:
            terminals[t.name.strip("'\"")] = t

    # Build parent map and depth map
    parents = {}
    depths = {}
    def _walk(node, depth=0.0):
        depths[id(node)] = depth
        for child in node.clades:
            bl = child.branch_length if child.branch_length is not None else 0.0
            parents[id(child)] = node
            _walk(child, depth + bl)
    _walk(tree_obj.root)

    def _path_to_root(clade):
        path = [id(clade)]
        curr = id(clade)
        while curr in parents:
            curr = id(parents[curr])
            path.append(curr)
        return path

    def _patristic_dist_to_tip(node, tip_clade):
        """Patristic distance from an internal node to a tip via LCA."""
        path_node = _path_to_root(node)
        path_tip = _path_to_root(tip_clade)
        path_node_set = set(path_node)
        # Find LCA
        lca_id = None
        for cid in path_tip:
            if cid in path_node_set:
                lca_id = cid
                break
        if lca_id is None:
            return 0.0
        # Distance from node to LCA
        d_node = depths[id(node)] - depths[lca_id]
        # Distance from tip to LCA
        d_tip = depths[id(tip_clade)] - depths[lca_id]
        return d_node + d_tip

    # Collect internal nodes (non-terminal clades)
    internal_nodes = []
    for clade in tree_obj.find_clades():
        if not clade.is_terminal():
            # Skip the root if it has only one child (degenerate)
            if clade is tree_obj.root and len(clade.clades) == 1:
                continue
            # Label the node
            label = clade.name or clade.confidence
            if label is None:
                # Generate a label from descendant leaves
                desc_leaves = [t.name.strip("'\"") for t in clade.get_terminals()
                               if t.name and t.name.strip("'\"") in taxa_set]
                label = f"node_{'-'.join(sorted(desc_leaves)[:3])}"
            internal_nodes.append((label, clade))

    results = []
    for label, clade in internal_nodes:
        # Get descendant tip indices
        desc_tip_names = [t.name.strip("'\"") for t in clade.get_terminals()
                          if t.name and t.name.strip("'\"") in taxa_set]
        desc_indices = [i for i, t in enumerate(taxa) if t in set(desc_tip_names)]

        # Compute patristic distances from this node to ALL tips
        n = len(taxa)
        node_dists = np.zeros(n, dtype=np.float64)
        for i, tname in enumerate(taxa):
            tip = terminals.get(tname)
            if tip is not None:
                node_dists[i] = _patristic_dist_to_tip(clade, tip)

        # MDS embed this node's distance vector to get a 4D position
        # We create a (n+1) x (n+1) distance matrix where the last row/col
        # is the internal node's distances to all tips, and the tip-by-tip
        # block is the existing distance matrix.
        # But simpler: just use the weighted centroid of descendant tips' z coords.
        if desc_indices:
            # Anchor = centroid of descendant tips only
            anchor = z_coords[desc_indices].mean(axis=0)
        else:
            anchor = np.zeros(z_coords.shape[-1], dtype=z_coords.dtype)

        results.append((str(label), anchor.astype(np.float32), desc_indices))

    return results


# ---------------------------------------------------------------------------
# Stage 1: base reconstruction from tips + patristic distances (no model)
# ---------------------------------------------------------------------------
def compute_centroid_tip_weights(
    z_coords: np.ndarray, lambda_dist: float, anchor: Optional[np.ndarray] = None,
    descendant_mask: Optional[np.ndarray] = None,
    node_tip_context_weight: float = 0.0,
) -> np.ndarray:
    """Weight each tip by proximity to the ancestral anchor point.

    By default the anchor is the MDS origin (the model's own [ROOT] placement,
    root-to-tip distance = ||z_n||). When an outgroup is supplied the anchor
    becomes the outgroup's MDS position, so the outgroup (distance 0) gets the
    highest weight and ingroup tips are weighted by proximity to it.

    In per-node mode, ``descendant_mask`` (a boolean array of length N) can be
    supplied to restrict contributions to descendant tips. The
    ``node_tip_context_weight`` controls how much non-descendant tips contribute:
    - 0.0 (default): hard mask, non-descendants get zero weight
    - >0.0: non-descendant weights are scaled by this factor (soft mask)
    - 1.0: no masking, all tips contribute equally (original behaviour)

    This is motivated by the observation that FastML's pruning algorithm uses
    all tips (weighted by branch length) for each node's likelihood, while a
    hard descendant mask discards non-descendant signal entirely. A soft mask
    allows non-descendants to contribute weak phylogenetic signal via the tree.

    z_coords: [N, 4] MDS coordinates.
    anchor:   [4] MDS position of the ancestor (default: origin).
    descendant_mask: [N] boolean, True for tips that are descendants of the node.
    node_tip_context_weight: scaling factor for non-descendant tip weights.
    Returns [N] normalised weights.
    """
    if anchor is None:
        anchor = np.zeros(z_coords.shape[-1], dtype=z_coords.dtype)
    root_dist = np.linalg.norm(z_coords - anchor, axis=-1)  # [N]
    w = np.exp(-lambda_dist * root_dist)
    if descendant_mask is not None and node_tip_context_weight < 1.0:
        scale = np.where(descendant_mask, 1.0, node_tip_context_weight)
        w = w * scale.astype(w.dtype)
    total = w.sum()
    return w / total if total > 0 else np.full_like(w, 1.0 / len(w))


def compute_base_logprobs(
    a_np: np.ndarray, tip_weights: np.ndarray, pseudocount: float
) -> np.ndarray:
    """Distance-weighted per-site amino-acid log-probabilities.

    a_np: [L, N] amino-acid tokens (0..19 valid, 20 = gap/unknown).
    tip_weights: [N] centroid proximity weights.
    Returns [L, 20] log-probabilities over the 20 standard amino acids.
    """
    L, N = a_np.shape
    logp = np.zeros((L, 20), dtype=np.float64)
    for i in range(L):
        row = a_np[i]
        valid = row < 20
        freqs = np.full(20, pseudocount, dtype=np.float64)
        if valid.any():
            for tok in range(20):
                freqs[tok] += tip_weights[valid][row[valid] == tok].sum()
        freqs /= freqs.sum()
        logp[i] = np.log(freqs)
    return logp


# ---------------------------------------------------------------------------
# Stage 2: HyphAeon-derived re-scoring terms
# ---------------------------------------------------------------------------
def compute_site_confidence(lrts: np.ndarray) -> np.ndarray:
    """Per-site reconstruction confidence w_conf(i) = 1 / (1 + LRT_i).

    Low selection (conserved / purifying) -> high confidence: the observed
    consensus is reliably ancestral. High diversifying selection -> low
    confidence: the ancestral state is genuinely ambiguous, so we down-weight
    the base term there and lean more on the joint/epistatic constraints.
    """
    return 1.0 / (1.0 + np.clip(lrts, 0.0, None))


def _append_ancestor_column(
    c_tensor: torch.Tensor,
    a_tensor: torch.Tensor,
    d_tensor: torch.Tensor,
    z_tensor: torch.Tensor,
    candidate_tokens: np.ndarray,
    anchor_z: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append the candidate ancestor as an extra taxon at the anchor position.

    By default the ancestor is placed at z = 0 (the MDS origin), with distance
    to each tip n equal to ||z_n|| — identical to the model's own [ROOT]
    placement. When ``anchor_z`` is supplied (outgroup rooting), the ancestor
    is placed at that position instead, with distance to tip n equal to
    ||z_n - anchor_z||.

    candidate_tokens: [L] amino-acid tokens (0..19; 20 -> gap) for the ancestor.
    anchor_z:         [4] MDS position for the ancestor (default: origin).
    """
    L, N, _ = c_tensor.shape
    anc_c = np.full((L, 1, 1), 64, dtype=np.int64)   # 64 = unknown/stop codon token
    anc_a = np.full((L, 1, 1), GAP_TOKEN, dtype=np.int64)
    for i in range(L):
        tok = int(candidate_tokens[i])
        if tok < 20:
            aa = REV_AA_MAP[tok]
            codon = CANONICAL_AA_TO_CODON[aa]
            anc_c[i, 0, 0] = GENETIC_CODE.get(codon, 64)
            anc_a[i, 0, 0] = AA_MAP.get(CODON_TO_AA.get(codon, "-"), GAP_TOKEN)

    c_ext = torch.cat([c_tensor, torch.from_numpy(anc_c)], dim=1)
    a_ext = torch.cat([a_tensor, torch.from_numpy(anc_a)], dim=1)

    z = z_tensor[0].cpu().numpy()                    # [N, 4]
    if anchor_z is None:
        anchor_z = np.zeros(z.shape[-1], dtype=z.dtype)
    anc_dist = np.linalg.norm(z - anchor_z, axis=-1)  # [N]
    d = d_tensor[0].cpu().numpy()                     # [N, N]
    d_ext = np.zeros((N + 1, N + 1), dtype=np.float32)
    d_ext[:N, :N] = d
    d_ext[N, :N] = anc_dist
    d_ext[:N, N] = anc_dist
    d_ext_t = torch.from_numpy(d_ext).unsqueeze(0)

    z_ext = np.zeros((N + 1, 4), dtype=np.float32)
    z_ext[:N] = z
    z_ext[N] = anchor_z                               # ancestor at anchor position
    z_ext_t = torch.from_numpy(z_ext).unsqueeze(0)

    return c_ext, a_ext, d_ext_t, z_ext_t


def compute_dms_deltas_for_candidate(
    model,
    c_tensor: torch.Tensor,
    a_tensor: torch.Tensor,
    d_tensor: torch.Tensor,
    z_tensor: torch.Tensor,
    taxa: List[str],
    candidate_tokens: np.ndarray,
    device: torch.device,
    progress: bool = False,
    anchor_z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run ancestor-focal Digital DMS and return per-site per-AA delta LRTs.

    Returns [L, 20] where entry [i, tok] is the selection-signal change from
    setting the ancestor's state at site i to amino acid `tok` (0 for the
    current wild-type state, by DMS construction).

    NOTE: the deltas are context-dependent on the ancestor's states at all OTHER
    sites, so they are only exact for the `candidate_tokens` passed in. The
    greedy search holds them fixed within a sweep and recomputes between sweeps
    (documented approximation).
    """
    anc_name = "__HYPHAEON_ANCESTOR__"
    c_ext, a_ext, d_ext, z_ext = _append_ancestor_column(
        c_tensor, a_tensor, d_tensor, z_tensor, candidate_tokens, anchor_z=anchor_z
    )
    tree_cache = model.precompute_tree_cache(d_ext.to(device), z_ext.to(device))
    results = run_insilico_selection_dms(
        model, c_ext, a_ext, tree_cache, taxa + [anc_name], device,
        focal_taxon=anc_name, progress=progress,
    )
    L = c_tensor.shape[0]
    deltas = np.zeros((L, 20), dtype=np.float64)
    for r in results:
        site = r["site"] - 1  # DMS reports 1-indexed sites
        for aa, d_lrt in r["mutant_deltas"].items():
            tok = AA_MAP.get(aa)
            if tok is not None:
                deltas[site, tok] = d_lrt
    return deltas


def _dms_regulariser(deltas: np.ndarray) -> np.ndarray:
    """Convert DMS delta LRTs into a soft penalty term [L, 20].

    delta_lrt is a sensitivity, not a fitness. We only penalise states that
    INCREASE the diversifying-selection signal (positive delta), leaving
    neutral / signal-reducing states unpenalised. This discourages ancestral
    candidates that would look anomalously "under selection" — a proxy for
    out-of-distribution / implausible states — without pretending the score is
    a likelihood.
    """
    return -np.clip(deltas, 0.0, None)


# ---------------------------------------------------------------------------
# Stage 2b: epistatic co-occurrence compatibility
# ---------------------------------------------------------------------------
def build_pairwise_cooccurrence(
    a_np: np.ndarray, edges: List[dict], pseudocount: float
) -> Dict[Tuple[int, int], np.ndarray]:
    """For each co-selected site pair, tabulate joint amino-acid co-occurrence.

    Returns {(i, j): [20, 20] log-frequency table} for i < j, both 0-indexed.
    Only pairs present in the co-selection network are considered — this is the
    epistatic constraint that standard site-independent ASR cannot express.
    """
    tables: Dict[Tuple[int, int], np.ndarray] = {}
    for e in edges:
        i = int(e["site_u"]) - 1
        j = int(e["site_v"]) - 1
        if i > j:
            i, j = j, i
        col_i, col_j = a_np[i], a_np[j]
        valid = (col_i < 20) & (col_j < 20)
        counts = np.full((20, 20), pseudocount, dtype=np.float64)
        for ti, tj in zip(col_i[valid], col_j[valid]):
            counts[ti, tj] += 1.0
        tables[(i, j)] = np.log(counts / counts.sum())
    return tables


# ---------------------------------------------------------------------------
# Stage 3: greedy coordinate-ascent search over candidate sequences
# ---------------------------------------------------------------------------
def _score_sequence(
    tokens: np.ndarray,
    base_logp: np.ndarray,
    confidence: np.ndarray,
    dms_penalty: np.ndarray,
    cooccur: Dict[Tuple[int, int], np.ndarray],
    lambda_plast: float,
    lambda_epi: float,
) -> Tuple[float, float, float, float]:
    """Total additive score = base + DMS regulariser + epistatic term."""
    idx = np.arange(len(tokens))
    base = float((confidence * base_logp[idx, tokens]).sum())
    dms = float(lambda_plast * dms_penalty[idx, tokens].sum())
    epi = 0.0
    for (i, j), table in cooccur.items():
        epi += table[tokens[i], tokens[j]]
    epi *= lambda_epi
    return base + dms + epi, base, dms, epi


def _greedy_optimise(
    init_tokens: np.ndarray,
    base_logp: np.ndarray,
    confidence: np.ndarray,
    dms_penalty: np.ndarray,
    cooccur: Dict[Tuple[int, int], np.ndarray],
    lambda_plast: float,
    lambda_epi: float,
    max_passes: int = 3,
) -> np.ndarray:
    """Coordinate ascent: greedily flip single sites to improve total score.

    DMS penalties are held fixed here (they are recomputed between search
    sweeps by the caller, since they depend on the full candidate context).
    """
    tokens = init_tokens.copy()
    idx = np.arange(len(tokens))
    # Sites participating in epistatic pairs, for fast neighbour lookup.
    partners: Dict[int, List[Tuple[int, int, int]]] = {}
    for k, (i, j) in enumerate(cooccur.keys()):
        partners.setdefault(i, []).append((i, j, 0))
        partners.setdefault(j, []).append((i, j, 1))

    for _ in range(max_passes):
        improved = False
        for site in range(len(tokens)):
            cur = tokens[site]
            # base + DMS contribution for this site alone
            base_dms = confidence[site] * base_logp[site] + lambda_plast * dms_penalty[site]
            # epistatic contribution as a function of this site's state
            epi_term = np.zeros(20, dtype=np.float64)
            for (i, j, pos) in partners.get(site, []):
                table = cooccur[(i, j)]
                other = tokens[j] if pos == 0 else tokens[i]
                epi_term += (table[:, other] if pos == 0 else table[other, :])
            candidate_scores = base_dms + lambda_epi * epi_term
            best = int(np.argmax(candidate_scores))
            if best != cur:
                tokens[site] = best
                improved = True
        if not improved:
            break
    return tokens


def run_ancestral_reconstruction(
    alignment_path: str,
    tree_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    variant: Optional[str] = None,
    use_tn93: bool = False,
    outgroup: Optional[str] = None,
    top_k: int = 5,
    lambda_dist: float = DEFAULT_LAMBDA_DIST,
    lambda_plast: float = DEFAULT_LAMBDA_PLAST,
    lambda_epi: float = DEFAULT_LAMBDA_EPI,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    search_sweeps: int = 3,
    cpu: bool = False,
    seed: int = 0,
    progress: bool = True,
    per_node: bool = False,
    node_tip_context_weight: float = 0.0,
    joint_pass: float = 0.0,
    weak_node_boost: float = 0.0,
) -> Dict:
    """End-to-end HyphAeon-informed ancestral candidate scoring.

    If ``outgroup`` is supplied (fuzzy taxon name), the ancestor is rooted at
    the outgroup's MDS position instead of the origin. The outgroup naturally
    dominates the tip weights (distance 0) and polarises derived vs ancestral
    states.

    Returns a dict with the ranked candidates and per-site diagnostics.
    """
    rng = np.random.default_rng(seed)
    device = get_device(cpu=cpu)

    # 1. Load alignment + patristic distances + MDS embedding.
    c_tensor, a_tensor, d_tensor, z_tensor, _inv, taxa, L = load_alignment_and_tree(
        alignment_path, tree_path, use_tn93=use_tn93
    )
    a_np = a_tensor.squeeze(-1).cpu().numpy()  # [L, N]
    z_coords = z_tensor[0].cpu().numpy()       # [N, 4]

    # 1b. Resolve outgroup -> anchor position in MDS space.
    anchor_z: Optional[np.ndarray] = None
    outgroup_name: Optional[str] = None
    outgroup_idx: Optional[int] = None
    if outgroup:
        for idx, t in enumerate(taxa):
            if outgroup.lower() in t.lower():
                outgroup_idx = idx
                outgroup_name = t
                break
        if outgroup_idx is None:
            raise ValueError(
                f"Outgroup '{outgroup}' not found among {len(taxa)} taxa."
            )
        anchor_z = z_coords[outgroup_idx].copy()
        if progress:
            print(f"[*] Outgroup rooting: '{outgroup_name}' (taxon {outgroup_idx + 1}/{len(taxa)})")

    # 1c. Determine the set of anchors to reconstruct.
    # In default mode: a single centroid ancestor (origin or outgroup position).
    # In per-node mode: one anchor per internal node of the tree.
    node_anchors: List[Tuple[str, np.ndarray]] = []
    if per_node:
        if use_tn93 or tree_path is None:
            raise ValueError(
                "Per-node reconstruction requires a tree. Provide -t/--tree and remove --no-tree."
            )
        from .dataset import extract_tree_from_string_or_file
        tree_obj = extract_tree_from_string_or_file(tree_path)
        if tree_obj is None:
            raise ValueError(f"Could not parse tree from '{tree_path}' for per-node mode.")
        internal_nodes = compute_internal_node_anchors(tree_obj, taxa, z_coords, outgroup_idx)
        node_anchors = [(label, az, di) for label, az, di in internal_nodes]
        if progress:
            print(f"[*] Per-node mode: {len(node_anchors)} internal nodes detected")
    else:
        node_anchors = [("centroid", anchor_z if anchor_z is not None else None, None)]

    # 2. HyphAeon selection LRTs + attributions -> confidence + co-selection net.
    # (Computed once; shared across all node anchors.)
    model = load_model(weights=weights_path, variant=variant, device=device)
    tree_cache = model.precompute_tree_cache(d_tensor.to(device), z_tensor.to(device))
    leaf_attr, lrts, _pvals, consensus_aas = compute_transformer_attributions(
        model, c_tensor, a_tensor, tree_cache, taxa, device, progress=progress
    )
    confidence = compute_site_confidence(lrts)
    edges, _graph = compute_branch_coselection_network(
        leaf_attr, lrts, taxa, consensus_aas
    )
    cooccur = build_pairwise_cooccurrence(a_np, edges, pseudocount)

    # 3. For each node anchor: base reconstruction + seed candidates + DMS/greedy search.
    all_node_results: List[Dict] = []
    for node_idx, (node_label, node_anchor, desc_indices) in enumerate(node_anchors):
        if progress and per_node:
            print(f"\n[*] Node {node_idx + 1}/{len(node_anchors)}: {node_label}")

        desc_mask = None
        if desc_indices is not None:
            desc_mask = np.zeros(len(taxa), dtype=bool)
            desc_mask[desc_indices] = True
        tip_weights = compute_centroid_tip_weights(z_coords, lambda_dist, anchor=node_anchor, descendant_mask=desc_mask, node_tip_context_weight=node_tip_context_weight)
        base_logp = compute_base_logprobs(a_np, tip_weights, pseudocount)

        # Weak-node boost: scale DMS/epistatic lambdas up for nodes with few descendants.
        node_lambda_plast = lambda_plast
        node_lambda_epi = lambda_epi
        if weak_node_boost > 0.0 and desc_indices is not None:
            n_desc = len(desc_indices)
            n_total = len(taxa)
            # Boost factor: 1.0 for root (all tips), up to (1 + weak_node_boost) for 2-tip nodes.
            # Scales linearly with inverse descendant fraction.
            signal_ratio = n_desc / n_total if n_total > 0 else 1.0
            boost = 1.0 + weak_node_boost * (1.0 - signal_ratio)
            node_lambda_plast = lambda_plast * boost
            node_lambda_epi = lambda_epi * boost

        consensus_tokens = np.argmax(base_logp, axis=1)
        seeds = [consensus_tokens]
        probs = np.exp(base_logp - base_logp.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        for _ in range(max(0, top_k - 1)):
            sampled = np.array([rng.choice(20, p=probs[i]) for i in range(L)])
            seeds.append(sampled)

        scored: List[AncestralCandidate] = []
        for s_i, seed_tokens in enumerate(seeds):
            tokens = seed_tokens.copy()
            for sweep in range(search_sweeps):
                dms_deltas = compute_dms_deltas_for_candidate(
                    model, c_tensor, a_tensor, d_tensor, z_tensor, taxa, tokens,
                    device, progress=False, anchor_z=node_anchor,
                )
                dms_penalty = _dms_regulariser(dms_deltas)
                new_tokens = _greedy_optimise(
                    tokens, base_logp, confidence, dms_penalty, cooccur,
                    node_lambda_plast, node_lambda_epi,
                )
                if np.array_equal(new_tokens, tokens):
                    break
                tokens = new_tokens
            total, base, dms, epi = _score_sequence(
                tokens, base_logp, confidence, dms_penalty, cooccur,
                node_lambda_plast, node_lambda_epi,
            )
            seq = "".join(REV_AA_MAP.get(int(t), "X") for t in tokens)
            scored.append(AncestralCandidate(
                sequence=seq, total_score=total, base_score=base,
                dms_score=dms, epi_score=epi,
                per_site_confidence=[float(x) for x in confidence],
            ))
            if progress:
                print(f"[*] Seed {s_i + 1}/{len(seeds)} -> score {total:.3f}", flush=True)

        unique: Dict[str, AncestralCandidate] = {}
        for cand in scored:
            prev = unique.get(cand.sequence)
            if prev is None or cand.total_score > prev.total_score:
                unique[cand.sequence] = cand
        ranked = sorted(unique.values(), key=lambda c: c.total_score, reverse=True)[:top_k]

        if per_node:
            all_node_results.append({
                "node_label": node_label,
                "candidates": [c.to_dict() for c in ranked],
            })
        else:
            # Single-node mode: return the original return format.
            return {
                "n_taxa": len(taxa),
                "n_sites": L,
                "n_coselection_edges": len(edges),
                "outgroup": outgroup_name,
                "candidates": [c.to_dict() for c in ranked],
                "per_site_confidence": [float(x) for x in confidence],
                "per_site_lrt": [float(x) for x in lrts],
            }

    # Per-node mode: optional joint consistency pass.
    if per_node and joint_pass > 0.0 and len(all_node_results) > 1:
        all_node_results = _joint_consistency_pass(
            node_anchors, all_node_results,
            a_np, z_coords, taxa, lambda_dist, pseudocount,
            model, c_tensor, a_tensor, d_tensor, z_tensor, device,
            confidence, cooccur, lambda_plast, lambda_epi,
            search_sweeps, top_k, rng, joint_pass, progress=progress,
        )

    # Per-node mode return format.
    return {
        "n_taxa": len(taxa),
        "n_sites": L,
        "n_coselection_edges": len(edges),
        "outgroup": outgroup_name,
        "per_node": True,
        "n_nodes": len(all_node_results),
        "nodes": all_node_results,
        "per_site_confidence": [float(x) for x in confidence],
        "per_site_lrt": [float(x) for x in lrts],
    }


def _joint_consistency_pass(
    node_anchors: List[Tuple[str, np.ndarray, List[int]]],
    all_node_results: List[Dict],
    a_np: np.ndarray,
    z_coords: np.ndarray,
    taxa: List[str],
    lambda_dist: float,
    pseudocount: float,
    model,
    c_tensor,
    a_tensor,
    d_tensor,
    z_tensor,
    device,
    confidence: np.ndarray,
    cooccur: Dict,
    lambda_plast: float,
    lambda_epi: float,
    search_sweeps: int,
    top_k: int,
    rng,
    joint_strength: float,
    progress: bool = False,
) -> List[Dict]:
    """Second pass: augment each node's base logprobs with a soft prior from
    its parent's reconstructed sequence.

    For each node, the parent's best candidate sequence is converted to a
    per-site log-probability prior (one-hot + pseudocount) and mixed with the
    node's own base logprobs at ``joint_strength`` weight. This introduces
    global consistency — a child node is gently pulled toward its parent's
    reconstruction, mimicking joint reconstruction.

    joint_strength: 0.0 = no effect (pure marginal), 1.0 = parent dominates.
    """
    # Build parent->children mapping from descendant set inclusion.
    # A node A is parent of node B if B's descendant set is a proper subset of A's.
    desc_sets = [set(di) for _, _, di in node_anchors]
    n_nodes = len(node_anchors)
    parent_of: List[Optional[int]] = [None] * n_nodes
    for child_idx in range(n_nodes):
        child_desc = desc_sets[child_idx]
        best_parent = None
        best_parent_size = float('inf')
        for par_idx in range(n_nodes):
            if par_idx == child_idx:
                continue
            par_desc = desc_sets[par_idx]
            if child_desc < par_desc and len(par_desc) < best_parent_size:
                best_parent = par_idx
                best_parent_size = len(par_desc)
        parent_of[child_idx] = best_parent

    if progress:
        print("\n[*] Joint consistency pass:")
        for i, pi in enumerate(parent_of):
            if pi is not None:
                print(f"    {node_anchors[i][0]} <- parent {node_anchors[pi][0]}")

    L = a_np.shape[0]
    updated_results: List[Dict] = []
    for node_idx, (node_label, node_anchor, desc_indices) in enumerate(node_anchors):
        if progress:
            print(f"\n[*] Joint pass — Node {node_idx + 1}/{n_nodes}: {node_label}")

        desc_mask = None
        if desc_indices is not None:
            desc_mask = np.zeros(len(taxa), dtype=bool)
            desc_mask[desc_indices] = True
        tip_weights = compute_centroid_tip_weights(z_coords, lambda_dist, anchor=node_anchor, descendant_mask=desc_mask)
        base_logp = compute_base_logprobs(a_np, tip_weights, pseudocount)

        # Mix in parent prior
        par_idx = parent_of[node_idx]
        if par_idx is not None and all_node_results[par_idx]["candidates"]:
            par_seq = all_node_results[par_idx]["candidates"][0]["sequence"]
            par_tokens = np.array([AA_MAP.get(aa, 20) for aa in par_seq], dtype=np.int64)
            parent_logp = np.full((L, 20), np.log(pseudocount), dtype=np.float64)
            for site in range(L):
                tok = par_tokens[site]
                if tok < 20:
                    parent_logp[site, tok] = 0.0  # log(1.0)
            base_logp = (1.0 - joint_strength) * base_logp + joint_strength * parent_logp

        consensus_tokens = np.argmax(base_logp, axis=1)
        seeds = [consensus_tokens]
        probs = np.exp(base_logp - base_logp.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        for _ in range(max(0, top_k - 1)):
            sampled = np.array([rng.choice(20, p=probs[i]) for i in range(L)])
            seeds.append(sampled)

        scored: List[AncestralCandidate] = []
        for s_i, seed_tokens in enumerate(seeds):
            tokens = seed_tokens.copy()
            for sweep in range(search_sweeps):
                dms_deltas = compute_dms_deltas_for_candidate(
                    model, c_tensor, a_tensor, d_tensor, z_tensor, taxa, tokens,
                    device, progress=False, anchor_z=node_anchor,
                )
                dms_penalty = _dms_regulariser(dms_deltas)
                new_tokens = _greedy_optimise(
                    tokens, base_logp, confidence, dms_penalty, cooccur,
                    lambda_plast, lambda_epi,
                )
                if np.array_equal(new_tokens, tokens):
                    break
                tokens = new_tokens
            total, base, dms, epi = _score_sequence(
                tokens, base_logp, confidence, dms_penalty, cooccur,
                lambda_plast, lambda_epi,
            )
            seq = "".join(REV_AA_MAP.get(int(t), "X") for t in tokens)
            scored.append(AncestralCandidate(
                sequence=seq, total_score=total, base_score=base,
                dms_score=dms, epi_score=epi,
                per_site_confidence=[float(x) for x in confidence],
            ))

        unique: Dict[str, AncestralCandidate] = {}
        for cand in scored:
            prev = unique.get(cand.sequence)
            if prev is None or cand.total_score > prev.total_score:
                unique[cand.sequence] = cand
        ranked = sorted(unique.values(), key=lambda c: c.total_score, reverse=True)[:top_k]
        updated_results.append({
            "node_label": node_label,
            "candidates": [c.to_dict() for c in ranked],
        })

    return updated_results


# ---------------------------------------------------------------------------
# CLI: python -m hyphaeon.ancestral -a aln.fasta [...]
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hyphaeon.ancestral",
        description="EXPERIMENTAL HyphAeon-informed ancestral sequence candidate scoring.",
    )
    p.add_argument("-a", "--alignment", required=True, help="In-frame codon alignment (FASTA/NEXUS)")
    p.add_argument("-t", "--tree", default=None, help="Optional Newick/NEXUS tree (embedded if omitted)")
    p.add_argument("--no-tree", action="store_true", help="Skip tree; use TN93 pairwise distances")
    p.add_argument("--use-tn93", action="store_true", help="Alias for --no-tree")
    p.add_argument("-w", "--weights", default=None, help="Local model weights (else HF download)")
    p.add_argument("--model-variant", dest="variant", default=None, help="HF model variant")
    p.add_argument("--outgroup", default=None, help="Outgroup taxon name (fuzzy match) to root the ancestor at its MDS position")
    p.add_argument("-k", "--top-k", type=int, default=5, help="Number of candidates to emit")
    p.add_argument("--lambda-dist", type=float, default=DEFAULT_LAMBDA_DIST,
                   help="Steepness of exponential distance weighting for tip contributions (default: %(default)s). "
                        "Lower values (e.g. 1.0-2.0) give a flatter weighting where distant/outgroup taxa contribute more; "
                        "higher values (e.g. 6.0-8.0) concentrate weight on the closest taxa. "
                        "When using a closely related outgroup, lower this to prevent the outgroup from dominating the reconstruction.")
    p.add_argument("--lambda-plast", type=float, default=DEFAULT_LAMBDA_PLAST)
    p.add_argument("--lambda-epi", type=float, default=DEFAULT_LAMBDA_EPI)
    p.add_argument("--sweeps", type=int, default=3, help="Max DMS/optimise alternation sweeps per seed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-node", action="store_true", help="Reconstruct one ancestor per internal tree node (requires -t/--tree)")
    p.add_argument("--node-tip-context-weight", type=float, default=0.0,
                   help="Non-descendant tip weight scaling in per-node mode (default: %(default)s). "
                        "0.0 = hard mask (descendants only), 1.0 = no mask (all tips contribute). "
                        "Intermediate values allow non-descendants to contribute weak phylogenetic signal.")
    p.add_argument("--joint-pass", type=float, default=0.0,
                   help="Joint consistency pass strength (default: %(default)s). "
                        "0.0 = disabled (pure marginal). >0.0 mixes each node's base logprobs with "
                        "a soft prior from its parent's reconstructed sequence. 1.0 = parent dominates.")
    p.add_argument("--weak-node-boost", type=float, default=0.0,
                   help="Boost DMS/epistatic lambdas for nodes with few descendants (default: %(default)s). "
                        "0.0 = uniform lambdas. >0.0 scales lambdas up to (1+boost) for 2-tip nodes, "
                        "linearly with inverse descendant fraction. Tests whether epistasis matters "
                        "most when phylogenetic signal is weak.")
    p.add_argument("--cpu", action="store_true", help="Force CPU execution")
    p.add_argument("-o", "--output", default=None, help="Optional JSON output path")
    p.add_argument("--fasta", default=None, help="Optional FASTA output path for candidates")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    use_tn93 = args.no_tree or args.use_tn93 or (args.tree == "tn93")
    if getattr(args, "per_node", False):
        if use_tn93:
            print("[!] --per-node requires a phylogenetic tree, but --no-tree/--use-tn93 was set.")
            print("    Provide -t/--tree and remove --no-tree/--use-tn93 to use per-node mode.")
            return 1
        if not args.tree:
            print("[!] --per-node requires a phylogenetic tree, but no tree was provided.")
            print("    Provide -t/--tree to use per-node mode.")
            return 1
    t0 = time.time()
    res = run_ancestral_reconstruction(
        alignment_path=args.alignment,
        tree_path=args.tree,
        weights_path=args.weights,
        variant=args.variant,
        use_tn93=use_tn93,
        outgroup=args.outgroup,
        top_k=args.top_k,
        lambda_dist=args.lambda_dist,
        lambda_plast=args.lambda_plast,
        lambda_epi=args.lambda_epi,
        search_sweeps=args.sweeps,
        cpu=args.cpu,
        seed=args.seed,
        per_node=getattr(args, "per_node", False),
        node_tip_context_weight=getattr(args, "node_tip_context_weight", 0.0),
        joint_pass=getattr(args, "joint_pass", 0.0),
        weak_node_boost=getattr(args, "weak_node_boost", 0.0),
    )
    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"Ancestral candidate scoring complete in {elapsed:.2f}s")
    print(f"  Taxa: {res['n_taxa']} | Sites: {res['n_sites']} | "
          f"Co-selection edges: {res['n_coselection_edges']}")
    if res.get("outgroup"):
        print(f"  Outgroup: {res['outgroup']}")
    if res.get("per_node"):
        print(f"  Per-node mode: {res['n_nodes']} internal nodes")
    print("=" * 72)

    if res.get("per_node"):
        for node_info in res["nodes"]:
            label = node_info["node_label"]
            best = node_info["candidates"][0]
            print(f"\n  [{label}]  best total={best['total_score']:.3f}  "
                  f"(base={best['base_score']:.3f}, dms={best['dms_score']:.3f}, "
                  f"epi={best['epi_score']:.3f})")
            print(f"    {best['sequence']}")
    else:
        for rank, cand in enumerate(res["candidates"], start=1):
            print(f"\n#{rank}  total={cand['total_score']:.3f}  "
                  f"(base={cand['base_score']:.3f}, dms={cand['dms_score']:.3f}, "
                  f"epi={cand['epi_score']:.3f})")
            print(f"    {cand['sequence']}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"\n[✓] JSON written to {args.output}")
    if args.fasta:
        with open(args.fasta, "w", encoding="utf-8") as fh:
            if res.get("per_node"):
                for node_info in res["nodes"]:
                    label = node_info["node_label"]
                    best = node_info["candidates"][0]
                    fh.write(f">{label} score={best['total_score']:.4f}\n")
                    fh.write(best["sequence"] + "\n")
            else:
                for rank, cand in enumerate(res["candidates"], start=1):
                    fh.write(f">ancestor_candidate_{rank} score={cand['total_score']:.4f}\n")
                    fh.write(cand["sequence"] + "\n")
        print(f"[✓] FASTA written to {args.fasta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
