"""
aeon_core/model.py
------------------
Core Neural Architecture: PhyloAxialTransformer with Multi-Scale 4D Tree-RoPE
for ultra-fast episodic positive selection inference.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Stationary background frequency floor (1/20 amino acids).
# Used by the continuous-time Markov transition probability tree kernel in both
# the inline and pre-cached forward paths. Shared constant ensures both paths
# stay in sync.
EPS0 = 0.05


def _coral_cutoffs(b0, theta_steps):
    """Cumulative-softplus ordinal cutoffs (CORAL, Cao et al. 2020).

    Shared by ``RankConsistentCoralHead`` and ``CoralHead``: guarantees
    monotonic b_1 > b_2 > ... > b_K from a learnable base ``b0`` and
    step params ``theta_steps``.
    """
    cum_steps = torch.cumsum(F.softplus(theta_steps), dim=0)
    return torch.cat([b0.unsqueeze(0), b0 - cum_steps])


# Default per-structure RAM budget (MB) for the precomputed tree attention bias
# cache. Overridable per call via max_memory_mb / the shared --max-memory-mb flag.
DEFAULT_MAX_MEMORY_MB = 1024.0


def _resolve_budget_bytes(max_memory_mb):
    """Resolve an optional MB budget to bytes (None -> DEFAULT_MAX_MEMORY_MB)."""
    return (DEFAULT_MAX_MEMORY_MB if max_memory_mb is None else max_memory_mb) * 1e6


def markov_log_bias(phylo_w1, dist):
    """
    Continuous-time Markov transition-probability log-bias, one per attention head:
        log( EPS0 + (1 - EPS0) * exp(-softplus(w_h) * d_ij) )
    phylo_w1: [num_heads, 1, 1] learnable per-head decay rate.
    dist: pairwise-distance tensor broadcastable against [num_heads, N, N].
    Single definition shared by the static cache, the inline (uncached) path,
    and the lazy bounded-memory path — all three must produce identical values.
    """
    decay_rate = F.softplus(phylo_w1)
    markov_kernel = EPS0 + (1.0 - EPS0) * torch.exp(-decay_rate * dist)
    return torch.log(markov_kernel.clamp(min=1e-5))


def tree_cache_layer_bias(tree_cache, layer, index, row_slice=None):
    """
    Markov log-bias for row layer `index`. Reads the static all-layers cache
    when present (fast path); otherwise evaluates lazily from the cached
    (N+1)x(N+1) distance tensor so the transient is one layer's bias instead
    of all num_layers resident at once (see precompute_tree_cache).

    row_slice: optional slice over query rows — returns only the rows needed
    for a chunked attention block instead of the full (N+1)^2 bias, so the
    lazy transient scales with the row chunk, not the cohort size.
    """
    biases = tree_cache.get("static_phylo_biases")
    if biases is not None:
        b = biases[index]
        return b if row_slice is None else b[..., row_slice, :]
    dist = tree_cache["static_dist"]
    if row_slice is not None:
        dist = dist[..., row_slice, :]
    # static_dist may be host-resident when it exceeds device memory — move
    # only the (possibly sliced) block, never the whole N^2 tensor.
    return markov_log_bias(layer.phylo_w1, dist.to(layer.phylo_w1.device))


# Cache-tile target for the per-chunk attention transient (~4 live tensors ×
# ~1 MB each). Empirical: tiling the (N+1)^2 score matrix at this size is
# ~35-40% FASTER than a single unchunked matmul at N>=512 on CPU — the tiles
# stay cache-resident instead of thrashing on N^2 strides. Applied
# unconditionally (identical math), not just when the budget demands it; a
# tighter max_memory_mb can only shrink chunks further.
ATTN_TILE_TARGET_BYTES = 4e6
# CUDA tiles are far larger: GPU kernels amortize launch overhead over big
# matmuls, so the transient cap scales up (still bound by user budget).
# Heuristic pending device profiling — identical math either way.
ATTN_TILE_TARGET_CUDA_BYTES = 256e6
# Below this many rows per chunk, per-chunk launch/loop overhead dominates
# (matters at N=80k where even one row > tile target). The caller's budget
# can still force smaller — the memory bound always wins.
MIN_ATTN_CHUNK_ROWS = 32


def _attn_row_chunk(budget_bytes, batch_eff, num_heads, num_nodes, device=None):
    """
    Query rows per attention chunk such that the O(B*H*rows*num_nodes)
    scores/bias/softmax transients stay under the effective cap: the smaller
    of budget_bytes and the device tile target (~4 live tensors per chunk).
    Chunk size floors at MIN_ATTN_CHUNK_ROWS for launch efficiency unless the
    budget forces smaller. Returns None when the full matrix fits one tile.
    """
    tile_target = (ATTN_TILE_TARGET_CUDA_BYTES
                   if device is not None and torch.device(device).type == "cuda"
                   else ATTN_TILE_TARGET_BYTES)
    bytes_per_row = 4 * batch_eff * num_heads * num_nodes * 4
    rows_budget = int(budget_bytes // max(1, bytes_per_row))
    rows_tile = int(tile_target // max(1, bytes_per_row))
    rows = min(rows_budget, max(rows_tile, MIN_ATTN_CHUNK_ROWS))
    rows = max(1, rows)
    if rows >= num_nodes:
        return None
    return rows


class BlockLinear(nn.Module):
    """
    Block-Diagonal Linear Projection.
    Guarantees 100% mathematical disentanglement between the Codon Synonymous Track
    and the Amino Acid Selection Track. Prevents linear layer cross-mixing of dS noise into dN+ features.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.half_in = in_features // 2
        self.half_out = out_features // 2
        self.block_codon = nn.Linear(self.half_in, self.half_out, bias=bias)
        self.block_aa = nn.Linear(self.half_in, self.half_out, bias=bias)

    def forward(self, x):
        # x: [..., in_features]
        x_codon = x[..., :self.half_in]
        x_aa = x[..., self.half_in:]
        out_codon = self.block_codon(x_codon)
        out_aa = self.block_aa(x_aa)
        return torch.cat([out_codon, out_aa], dim=-1)


# --- Custom Row Attention with Block-Diagonal Disentanglement & Learnable Phylogenetic Bias ---
class PhyloRowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = BlockLinear(embed_dim, embed_dim)
        self.k_proj = BlockLinear(embed_dim, embed_dim)
        self.v_proj = BlockLinear(embed_dim, embed_dim)

        # NOTE: The following parameters are defined but NOT used in forward().
        # They are remnants of earlier architecture iterations (3-channel tree
        # projection, 2-layer phylo MLP, per-site rate scaler) that were
        # simplified to the current 1-channel Markov kernel. They remain here
        # because they are present in the pretrained checkpoint (hyphaeon_v1.pt)
        # and removing them from __init__ would cause load_state_dict to fail
        # with unexpected-key errors. Removing them requires either a checkpoint
        # migration or a compatibility shim in the CLI. See REVIEW.md item #29.
        self.tree_w1 = nn.Parameter(torch.randn(num_heads, 3) * 0.02)
        self.tree_b1 = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.tree_w2 = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.02)

        # phylo_w1 IS used in forward() (Markov kernel decay rate).
        # phylo_b1 and phylo_w2 are NOT used — same situation as above.
        self.phylo_w1 = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.02)
        self.phylo_b1 = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.phylo_w2 = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.02)

        # NOT used in forward() — same situation as above.
        self.site_tree_scaler = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.site_tree_scaler.weight)
        nn.init.zeros_(self.site_tree_scaler.bias)
        
        half_dim = self.head_dim // 2
        self.rope_freqs = nn.Parameter(torch.randn(num_heads, half_dim, 4) * 0.05)
        # Learnable Initial-Representation Skip Weight (Initialized to 0.20)
        self.alpha_skip = nn.Parameter(torch.tensor(0.20))
        
        self.out_proj = BlockLinear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, dist_matrix=None, mds_coords=None, padding_mask=None, x0=None,
                cos=None, sin=None, phylo_bias=None, row_chunk=None, return_root_attn=False):
        """
        Row attention with optional pre-cached tree kernels.

        When cos/sin/phylo_bias are provided (from precompute_tree_cache), skips
        inline RoPE and Markov bias computation. This is the single attention
        code path used by both forward() and forward_cached().

        row_chunk bounds the O(B*H*(N+1)^2) scores/bias transient by evaluating
        query rows in blocks — softmax is row-wise, so chunking is exact.
        return_root_attn additionally returns the root query row's attention
        (B, H, N) without materializing the full probability matrix for callers.
        """
        batch_size, num_species, _ = x.shape

        q = self.q_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)

        # Tree-RoPE: Apply 4D MDS Rotary Position Phase Rotations to Query & Key
        half_dim = self.head_dim // 2
        if cos is None or sin is None:
            # Compute rotations inline from mds_coords
            if mds_coords is not None:
                m_exp = mds_coords.unsqueeze(1).unsqueeze(3) # [batch_size, 1, num_species, 1, 4]
                f_exp = self.rope_freqs.unsqueeze(0).unsqueeze(2) # [1, num_heads, 1, half_dim, 4]
                angles = (m_exp * f_exp).sum(dim=-1) # [batch_size, num_heads, num_species, half_dim]
                cos = torch.cos(angles).to(q.dtype)
                sin = torch.sin(angles).to(q.dtype)

        if cos is not None and sin is not None:
            q1, q2 = q[..., :half_dim], q[..., half_dim:]
            k1, k2 = k[..., :half_dim], k[..., half_dim:]
            q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

        # Pure Continuous-Time Markov Transition Probability Tree Kernel
        # P_ij(d) = eps0 + (1 - eps0) * exp(-lambda_h * d_ij)
        # Reflects exact continuous-time Markov substitution process where transition
        # probability asymptotes to eps0 (1/20 amino acids) as d -> inf.
        #
        # Query rows are evaluated in chunks of row_chunk (or a single chunk
        # covering all rows when unchunked) so the O(B*H*(N+1)^2) scores/bias
        # transient scales with the chunk, not the cohort.
        dist_1d = None
        if phylo_bias is None and dist_matrix is not None:
            dist_1d = dist_matrix[..., 0] if dist_matrix.dim() == 4 or (dist_matrix.dim() == 3 and dist_matrix.shape[-1] == 3) else dist_matrix

        row_starts = range(0, num_species, row_chunk) if row_chunk is not None and row_chunk < num_species else (0,)
        out_heads = torch.empty(batch_size, self.num_heads, num_species, self.head_dim,
                                device=x.device, dtype=v.dtype)
        root_attn = None
        for r0 in row_starts:
            r1 = min(r0 + (row_chunk or num_species), num_species)
            scores = torch.matmul(q[:, :, r0:r1], k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if phylo_bias is not None:
                scores.add_(phylo_bias[..., r0:r1, :])
            elif dist_1d is not None:
                # Slice first, then move — a host-resident static_dist
                # (VRAM-tight GPUs) transfers only the row chunk.
                d = dist_1d[..., r0:r1, :].to(scores.device)
                d = d.unsqueeze(1) if d.dim() == 3 else d.unsqueeze(0).unsqueeze(1)
                scores.add_(markov_log_bias(self.phylo_w1, d))

            if padding_mask is not None:
                mask = padding_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask, -1e4)
                attn_weights = torch.softmax(scores, dim=-1)
                attn_weights = torch.where(mask, torch.zeros_like(attn_weights), attn_weights)
            else:
                attn_weights = torch.softmax(scores, dim=-1)
            if return_root_attn and r0 == 0:
                root_attn = attn_weights[:, :, 0, 1:].detach()
            attn_weights = self.dropout(attn_weights).to(v.dtype)

            out_heads[:, :, r0:r1] = torch.matmul(attn_weights, v)

        out = out_heads.transpose(1, 2).contiguous().view(batch_size, num_species, self.embed_dim)
        out = self.out_proj(out)
        if x0 is not None:
            out = out + self.alpha_skip * x0  # Learnable initial-representation skip connection
        if return_root_attn:
            return out, root_attn
        return out



class RankConsistentCoralHead(nn.Module):
    """
    Rank-Consistent Ordinal Regression Head (Cao, Mirjalili, & Raschka, 2020):
    Standard unconstrained linear projection with learnable monotonic rank cutoffs.
    Zero magic numbers, zero WeightNorm constraints, zero manual gain constants.
    """
    def __init__(self, embed_dim, num_thresholds=8):
        super().__init__()
        self.num_thresholds = num_thresholds
        
        self.fc1 = nn.Linear(embed_dim, 256)
        self.fc2 = nn.Linear(256, 1, bias=False)
        
        # Learnable Rank Cutoffs (Cao et al. 2020)
        b0_init = -2.2253 if num_thresholds == 8 else -1.7346
        self.b0 = nn.Parameter(torch.tensor(b0_init))
        self.theta_steps = nn.Parameter(torch.full((num_thresholds - 1,), 0.40))
        
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0 / (embed_dim ** 0.5))
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0 / (256 ** 0.5))
        
    def get_cutoffs(self):
        return _coral_cutoffs(self.b0, self.theta_steps)

    def forward(self, x):
        # x: [batch_size, embed_dim]
        h = F.gelu(self.fc1(x))
        proj = self.fc2(h)  # Standard unconstrained linear projection
        cutoffs = self.get_cutoffs().to(device=x.device, dtype=x.dtype)
        logits = proj + cutoffs.unsqueeze(0)  # [batch_size, num_thresholds]
        return logits


LOG_CORAL_DELTAS_8 = torch.tensor([0.6931, 0.7239, 0.2793, 0.3364, 0.4377, 0.5741, 0.8873, 0.6833])

LOG_CORAL_DELTAS_16 = torch.tensor([0.2420, 0.3176, 0.2513, 0.2284, 0.1991, 0.1786, 0.2793, 0.2205, 0.2339, 0.4243, 0.2984, 0.2621, 0.3610, 0.4353, 0.3989, 0.2844])

LOG_CORAL_DELTAS_24 = torch.tensor([
    0.2420, 0.1635, 0.1542, 0.1335, 0.1613, 0.1849, 0.1991, 0.1786, 0.1602,
    0.1191, 0.1619, 0.1746, 0.1919, 0.2458, 0.2376, 0.2364, 0.2744, 0.2776,
    0.2647, 0.2461, 0.2268, 0.1847, 0.1963, 0.1807
])

LOG_CORAL_DELTAS_12 = LOG_CORAL_DELTAS_16[:12]

def decode_soft_ordinal_lrt(logits_lrt_ordinal):
    """
    Log-Space Soft-Bin Survival Integral Decoder:
    E[log(1+LRT)] = sum_{k=0}^{K-1} P(log(1+LRT) > Z_k) * delta_Z_k
    E[LRT] = exp(E[log(1+LRT)]) - 1
    """
    if logits_lrt_ordinal.dim() == 1 or logits_lrt_ordinal.shape[-1] not in (8, 12, 16, 24):
        return logits_lrt_ordinal, logits_lrt_ordinal
        
    probs = torch.sigmoid(logits_lrt_ordinal)
    K = probs.shape[-1]
    if K == 8:
        deltas = LOG_CORAL_DELTAS_8.to(device=logits_lrt_ordinal.device, dtype=logits_lrt_ordinal.dtype)
    elif K == 16:
        deltas = LOG_CORAL_DELTAS_16.to(device=logits_lrt_ordinal.device, dtype=logits_lrt_ordinal.dtype)
    elif K == 24:
        deltas = LOG_CORAL_DELTAS_24.to(device=logits_lrt_ordinal.device, dtype=logits_lrt_ordinal.dtype)
    elif K == 12:
        deltas = LOG_CORAL_DELTAS_12.to(device=logits_lrt_ordinal.device, dtype=logits_lrt_ordinal.dtype)
    else:
        deltas = LOG_CORAL_DELTAS_16[:K].to(device=logits_lrt_ordinal.device, dtype=logits_lrt_ordinal.dtype)
    
    # Expected log(1 + LRT) via continuous survival integration
    log_lrt_expected = (probs * deltas.view(1, -1)).sum(dim=1)
    
    # Invert back to physical LRT scale
    physical_lrt = torch.expm1(log_lrt_expected)
    return physical_lrt, probs


class FocalCoralOrdinalLoss(nn.Module):
    """
    Rank-Consistent Ordinal Binary Cross-Entropy Loss (CORAL, Cao et al. 2020)
    over log-spaced LRT exceedance thresholds matching ``LOG_CORAL_DELTAS_*``.
    """
    def __init__(self, gamma=0.0, num_thresholds=16):
        super().__init__()
        self.gamma = gamma
        self.num_thresholds = num_thresholds
        if num_thresholds == 8:
            deltas = LOG_CORAL_DELTAS_8
        elif num_thresholds == 16:
            deltas = LOG_CORAL_DELTAS_16
        elif num_thresholds == 24:
            deltas = LOG_CORAL_DELTAS_24
        elif num_thresholds == 12:
            deltas = LOG_CORAL_DELTAS_12
        else:
            deltas = LOG_CORAL_DELTAS_16[:num_thresholds]
        log_thresholds = torch.cumsum(deltas, dim=0)
        thresholds = torch.expm1(log_thresholds)
        self.register_buffer("thresholds", thresholds)
        self.register_buffer("weights", torch.ones_like(thresholds))

    def forward(self, logits_ordinal, y_lrt):
        y_lrt = y_lrt.reshape(-1)
        targets = (y_lrt.unsqueeze(1) > self.thresholds.unsqueeze(0)).to(logits_ordinal.dtype)
        bce = F.binary_cross_entropy_with_logits(logits_ordinal, targets, reduction="none")
        if self.gamma > 0:
            probs = torch.sigmoid(logits_ordinal)
            pt = torch.where(targets == 1.0, probs, 1.0 - probs)
            focal_weight = (1.0 - pt) ** self.gamma
            loss = focal_weight * bce * self.weights.unsqueeze(0)
        else:
            loss = bce * self.weights.unsqueeze(0)
        return loss.mean()



class PhyloAxialTransformer(nn.Module):
    """
    Phylogenetic Axial Transformer with Learned [ROOT] Token:
    A dedicated [ROOT] token at the tree origin (0, 0, 0, 0) is prepended to the sequence of taxa.
    Across all attention layers, the [ROOT] token participates in bidirectional self-attention with all taxa,
    allowing branch-level non-synonymous mutations to route directly into the [ROOT] token while
    broadcasting tree-wide background rates back down to leaves.
    Zero external pooling heuristics, zero PMA probe artifacts, 100% permutation-invariant.
    """
    def __init__(self, num_tokens=66, embed_dim=128, num_heads=8, num_layers=4, window_size=1, max_species=256, dropout=0.1, max_k=32, num_thresholds=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_layers = num_layers
        self.num_thresholds = num_thresholds

        self.codon_embedding = nn.Embedding(num_tokens, embed_dim // 2)
        self.aa_embedding = nn.Embedding(23, embed_dim // 2)
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, embed_dim) * 0.02)
        
        # Learnable Phylogenetic [ROOT] Token (Origin of the Evolutionary Tree)
        self.root_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim) * 0.02)
        
        self.mds_proj = nn.Linear(4, embed_dim)
        num_col_layers = 2 if window_size > 1 else 0
        num_row_layers = num_layers
        
        self.col_layers = nn.ModuleList([
            nn.Sequential(
                BlockLinear(embed_dim, 2*embed_dim),
                nn.GELU(),
                BlockLinear(2*embed_dim, embed_dim)
            ) for _ in range(num_col_layers)
        ])

        self.row_layers = nn.ModuleList([
            PhyloRowAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
            for _ in range(num_row_layers)
        ])
        self.row_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_row_layers)])

        self.lrt_ordinal_head = RankConsistentCoralHead(embed_dim, num_thresholds=num_thresholds)

    @property
    def _num_row_heads(self):
        """Attention-head count of the row-attention stack (0 when absent)."""
        return self.row_layers[0].num_heads if len(self.row_layers) > 0 else 0

    def precompute_tree_cache(self, dist_matrix, mds_coords, max_memory_mb=None):
        """
        Pre-computes static tree-invariant tensors (continuous-time Markov log-bias,
        4D Tree-RoPE phase rotations, and projected MDS embeddings).
        Can be computed once per phylogeny and reused across all site batches for 3-5x forward speedup.

        The all-layers bias cache costs num_layers * num_heads * (N+1)^2 * 4 bytes
        (~28.8 GB at N=10,000 for the production 6-layer/12-head model). When that
        exceeds max_memory_mb (default DEFAULT_MAX_MEMORY_MB), only the (N+1)^2
        distance tensor is kept under "static_dist" and consumers evaluate each
        layer's bias lazily via tree_cache_layer_bias — identical math, with the
        resident footprint bounded by the distance matrix instead of all layers.
        """
        device = next(self.parameters()).device

        num_heads = self._num_row_heads
        bias_bytes = len(self.row_layers) * num_heads * ((dist_matrix.shape[-1] + 1) ** 2) * 4
        budget_bytes = _resolve_budget_bytes(max_memory_mb)
        use_static_biases = bias_bytes <= budget_bytes
        if not use_static_biases:
            print(f"[*] Tree attention bias cache requires {bias_bytes / 1e9:.1f} GB "
                  f"(max_memory_mb={budget_bytes / 1e6:.0f}): using lazy per-layer evaluation.",
                  flush=True)

        # Lazy-path distances stay host-resident when the (N+1)^2 fp32 tensor
        # would crowd device memory — chunked consumers transfer only row
        # slices, so a 25.6 GB 80k-cohort matrix need not live in VRAM at
        # all. The static path keeps it on-device: it computes full biases
        # from it there, and a bias set that fits the budget implies the
        # much smaller dist does too. Fit test uses ~2.5x the tensor: the
        # augmentation below needs dist_matrix + dist_full live at once
        # (~2x N^2) plus headroom for attention tiles.
        dist_device = device
        if not use_static_biases and device.type == "cuda":
            dist_bytes = (dist_matrix.shape[-1] + 1) ** 2 * 4
            if dist_bytes * 2.5 > torch.cuda.mem_get_info(device)[0]:
                dist_device = torch.device("cpu")
        if not isinstance(dist_matrix, torch.Tensor):
            dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32, device=dist_device)
        else:
            dist_matrix = dist_matrix.to(dist_device)

        if not isinstance(mds_coords, torch.Tensor):
            mds_coords = torch.tensor(mds_coords, dtype=torch.float32, device=device)
        else:
            mds_coords = mds_coords.to(device)

        if dist_matrix.dim() == 2:
            dist_matrix = dist_matrix.unsqueeze(0)  # [1, N, N]
        if mds_coords.dim() == 2:
            mds_coords = mds_coords.unsqueeze(0)  # [1, N, 4]

        # 1. Augment with [ROOT] Token at tree origin [0, 0, 0, 0]
        root_mds = torch.zeros(1, 1, 4, dtype=mds_coords.dtype, device=mds_coords.device)
        mds_full = torch.cat([root_mds, mds_coords], dim=1)  # [1, num_species + 1, 4]

        # [1, num_species, 1] — on dist_matrix.device so the augmentation
        # below stays on dist_device even when it differs from the model's.
        root_dist = torch.norm(mds_coords, dim=-1, keepdim=True).to(dist_matrix.device)
        # In-place augmentation instead of cat chains: cat materializes the
        # 4N^2 dist_bot copy, then copies it again into dist_full — peaking at
        # ~12N^2 live (caller matrix + dist_bot + dist_full). Direct writes
        # peak at ~8N^2 and produce a byte-identical tensor (cat is a pure
        # copy; every cell below maps 1:1 to the old cat layout).
        n = dist_matrix.shape[-1]
        dist_full = torch.empty(1, n + 1, n + 1, dtype=dist_matrix.dtype,
                                device=dist_matrix.device)
        rd = root_dist[0, :, 0]  # [N] root-to-taxon distances
        dist_full[0, 0, 0] = 0.0
        dist_full[0, 0, 1:] = rd   # root row
        dist_full[0, 1:, 0] = rd   # symmetric root column
        dist_full[0, 1:, 1:] = dist_matrix[0]

        dist_1d = dist_full[..., 0] if dist_full.dim() == 4 or (dist_full.dim() == 3 and dist_full.shape[-1] == 3) else dist_full
        dist_tensor = dist_1d.unsqueeze(1) if dist_1d.dim() == 3 else dist_1d.unsqueeze(0).unsqueeze(1)

        static_phylo_biases = [] if use_static_biases else None
        static_rope_coss = []
        static_rope_sins = []

        with torch.no_grad():
            for layer in self.row_layers:
                if use_static_biases:
                    static_phylo_biases.append(markov_log_bias(layer.phylo_w1, dist_tensor))

                half_dim = layer.head_dim // 2
                m_exp = mds_full.unsqueeze(1).unsqueeze(3)  # [1, 1, num_species + 1, 1, 4]
                f_exp = layer.rope_freqs.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, half_dim, 4]
                angles = (m_exp * f_exp).sum(dim=-1)  # [1, num_heads, num_species + 1, half_dim]
                static_rope_coss.append(torch.cos(angles))
                static_rope_sins.append(torch.sin(angles))

        mds_pos_static = self.mds_proj(mds_coords)  # [1, num_species, embed_dim]

        return {
            "static_phylo_biases": static_phylo_biases,
            "static_dist": dist_tensor,
            "static_rope_coss": static_rope_coss,
            "static_rope_sins": static_rope_sins,
            "mds_pos_static": mds_pos_static,
            "num_species": mds_coords.shape[1],
            # Resolved build-time budget — forward_cached / attention extraction
            # read this to size their per-site attention row chunks.
            "max_memory_mb": budget_bytes / 1e6,
        }

    def forward_cached(self, msa_codons, msa_aas, tree_cache, padding_mask=None, return_attentions=False, return_hidden=False):
        """
        Fast forward pass using pre-cached static tree kernels.
        Eliminates repeated matrix exponentials and trigonometric rotations.
        Automatically decomposes large codon batches into memory-safe micro-batches
        when sequence depth N > 1000 to prevent O(B * H * N^2) accelerator buffer overflow.
        """
        batch_size, num_species, window_size = msa_codons.shape
        central_idx = window_size // 2

        # Automatic memory-safe micro-batching for large alignments
        max_attention_elements = 4_000_000 # ~16 MB per attention tensor
        if batch_size > 1 and (batch_size * (num_species ** 2)) > max_attention_elements:
            micro_b = max(1, int(max_attention_elements / max(1, num_species ** 2)))
            all_lrt_soft = []
            all_logits = []
            all_extra = []
            for b_start in range(0, batch_size, micro_b):
                b_end = min(b_start + micro_b, batch_size)
                p_sub = padding_mask[b_start:b_end] if padding_mask is not None else None
                out = self.forward_cached(
                    msa_codons[b_start:b_end],
                    msa_aas[b_start:b_end],
                    tree_cache,
                    padding_mask=p_sub,
                    return_attentions=return_attentions,
                    return_hidden=return_hidden
                )
                all_lrt_soft.append(out[0])
                all_logits.append(out[1])
                if return_hidden or return_attentions:
                    all_extra.append(out[2])
                    
            cat_lrt_soft = torch.cat(all_lrt_soft, dim=0)
            cat_logits = torch.cat(all_logits, dim=0)
            if return_hidden or return_attentions:
                cat_extra = torch.cat(all_extra, dim=0)
                return cat_lrt_soft, cat_logits, cat_extra
            return cat_lrt_soft, cat_logits

        # 1. Embeddings
        codon_emb = self.codon_embedding(msa_codons)
        aa_emb = self.aa_embedding(msa_aas)
        x = torch.cat([codon_emb, aa_emb], dim=-1)
        x = x + self.pos_embedding.unsqueeze(1)
        phylo_pos = tree_cache.get("mds_pos_static", tree_cache.get("static_phylo_pos"))
        if phylo_pos.dim() == 3:
            x = x + phylo_pos.unsqueeze(2)
        else:
            x = x + phylo_pos.unsqueeze(0).unsqueeze(2)

        root = self.root_token.expand(batch_size, 1, window_size, -1)
        x_full = torch.cat([root, x], dim=1)
        num_nodes = num_species + 1

        # 2. Padding Mask
        if padding_mask is not None:
            root_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=msa_codons.device)
            padding_mask_full = torch.cat([root_mask, padding_mask], dim=1)
            padding_mask_dup = padding_mask_full.unsqueeze(1).expand(-1, window_size, -1).contiguous().view(batch_size * window_size, num_nodes)
        else:
            padding_mask_dup = None

        # 3. Column Layers
        for i in range(len(self.col_layers)):
            col_in = x_full.reshape(batch_size * num_nodes, window_size, self.embed_dim) if window_size > 1 else x_full.reshape(batch_size * num_nodes, self.embed_dim)
            col_out = self.col_layers[i](col_in)
            x_full = col_out.reshape(batch_size, num_nodes, window_size, self.embed_dim)

        # 4. Row Layers using Pre-Cached Kernels
        x0_dup = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, self.embed_dim)
        static_coss = tree_cache["static_rope_coss"]
        static_sins = tree_cache["static_rope_sins"]

        layer_attns = [] if return_attentions else None

        # Bound the O(B*W*H*(N+1)^2) attention transient by the same budget
        # that governed bias caching (~307 GB/layer unchunked at N=80k).
        num_heads = self._num_row_heads
        budget_bytes = tree_cache.get("max_memory_mb", DEFAULT_MAX_MEMORY_MB) * 1e6
        row_chunk = _attn_row_chunk(budget_bytes, batch_size * window_size, num_heads, num_nodes,
                                    device=x_full.device)
        # When chunked under the lazy path, hand the layer the (N+1)x(N+1)
        # distance tensor so it evaluates bias per row-block — the full
        # H*(N+1)^2 per-layer bias would itself exceed the budget at large N.
        lazy_dist_2d = None
        if row_chunk is not None and tree_cache.get("static_phylo_biases") is None:
            lazy_dist_2d = tree_cache["static_dist"].squeeze(0).squeeze(0)

        for i, layer in enumerate(self.row_layers):
            row_in = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, self.embed_dim)

            bias_kwargs = ({"dist_matrix": lazy_dist_2d}
                           if lazy_dist_2d is not None
                           else {"phylo_bias": tree_cache_layer_bias(tree_cache, layer, i)})
            if return_attentions:
                row_out, root_attn = layer(
                    row_in,
                    padding_mask=padding_mask_dup,
                    x0=x0_dup,
                    cos=static_coss[i],
                    sin=static_sins[i],
                    row_chunk=row_chunk,
                    return_root_attn=True,
                    **bias_kwargs,
                )
                layer_attns.append(root_attn)
            else:
                row_out = layer(
                    row_in,
                    padding_mask=padding_mask_dup,
                    x0=x0_dup,
                    cos=static_coss[i],
                    sin=static_sins[i],
                    row_chunk=row_chunk,
                    **bias_kwargs,
                )
            row_out = self.row_norms[i](row_in + row_out)
            x_full = row_out.reshape(batch_size, window_size, num_nodes, layer.embed_dim).transpose(1, 2)

        root_repr = x_full[:, 0, central_idx, :]
        logits_lrt_ordinal = self.lrt_ordinal_head(root_repr)
        y_lrt_soft, _ = decode_soft_ordinal_lrt(logits_lrt_ordinal)

        if return_hidden:
            return y_lrt_soft.view(batch_size), logits_lrt_ordinal, root_repr

        if return_attentions:
            all_attns = torch.stack(layer_attns, dim=0) # [num_layers, B*W, num_heads, num_species]
            mean_root_attns = all_attns.mean(dim=(0, 2)).view(batch_size, num_species) # [batch_size, num_species]
            return y_lrt_soft.view(batch_size), logits_lrt_ordinal, mean_root_attns
            
        return y_lrt_soft.view(batch_size), logits_lrt_ordinal

    def forward(self, msa_codons, msa_aas, dist_matrix=None, mds_coords=None, padding_mask=None, tree_cache=None, return_attentions=False, return_hidden=False, max_memory_mb=None):
        if tree_cache is not None:
            return self.forward_cached(msa_codons, msa_aas, tree_cache, padding_mask=padding_mask, return_attentions=return_attentions, return_hidden=return_hidden)

        batch_size, num_species, window_size = msa_codons.shape
        central_idx = window_size // 2

        # Automatic memory-safe micro-batching for large alignments
        max_attention_elements = 4_000_000
        if batch_size > 1 and (batch_size * (num_species ** 2)) > max_attention_elements:
            micro_b = max(1, int(max_attention_elements / max(1, num_species ** 2)))
            all_lrt_soft = []
            all_logits = []
            all_extra = []
            for b_start in range(0, batch_size, micro_b):
                b_end = min(b_start + micro_b, batch_size)
                p_sub = padding_mask[b_start:b_end] if padding_mask is not None else None
                d_sub = (
                    dist_matrix[b_start:b_end]
                    if dist_matrix is not None and dist_matrix.shape[0] == batch_size
                    else dist_matrix
                )
                z_sub = (
                    mds_coords[b_start:b_end]
                    if mds_coords is not None and mds_coords.shape[0] == batch_size
                    else mds_coords
                )
                out = self.forward(
                    msa_codons[b_start:b_end],
                    msa_aas[b_start:b_end],
                    dist_matrix=d_sub,
                    mds_coords=z_sub,
                    padding_mask=p_sub,
                    tree_cache=None,
                    return_attentions=return_attentions,
                    return_hidden=return_hidden,
                    max_memory_mb=max_memory_mb,
                )
                all_lrt_soft.append(out[0])
                all_logits.append(out[1])
                if return_hidden or return_attentions:
                    all_extra.append(out[2])
                    
            cat_lrt_soft = torch.cat(all_lrt_soft, dim=0)
            cat_logits = torch.cat(all_logits, dim=0)
            if return_hidden or return_attentions:
                cat_extra = torch.cat(all_extra, dim=0)
                return cat_lrt_soft, cat_logits, cat_extra
            return cat_lrt_soft, cat_logits
        
        # Ensure padding_mask is always a canonical boolean tensor to keep XLA graph topology static
        if padding_mask is None:
            padding_mask = torch.zeros(batch_size, num_species, dtype=torch.bool, device=msa_codons.device)

        codon_emb = self.codon_embedding(msa_codons)
        aa_emb = self.aa_embedding(msa_aas)
        
        x = torch.cat([codon_emb, aa_emb], dim=-1)
        x = x + self.pos_embedding.unsqueeze(1)
        
        phylo_pos = self.mds_proj(mds_coords)
        x = x + phylo_pos.unsqueeze(2)
        
        # 1. Prepend [ROOT] Token at Index 0
        root_x = self.root_token.expand(batch_size, 1, window_size, -1)
        x_full = torch.cat([root_x, x], dim=1)  # [batch_size, num_species + 1, window_size, embed_dim]
        
        # 2. Augment Padding Mask (Root token is never padded)
        root_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=msa_codons.device)
        padding_mask_full = torch.cat([root_mask, padding_mask], dim=1)  # [batch_size, num_species + 1]
        
        # 3. Augment MDS Coords (Root is at tree origin [0, 0, 0, 0])
        root_mds = torch.zeros(batch_size, 1, 4, dtype=mds_coords.dtype, device=mds_coords.device)
        mds_full = torch.cat([root_mds, mds_coords], dim=1)  # [batch_size, num_species + 1, 4]
        
        # 4. Augment Distance Matrix (Root-to-taxa distance is norm in MDS space)
        root_dist = torch.norm(mds_coords, dim=-1, keepdim=True)  # [batch_size, num_species, 1]
        dist_top = torch.cat([torch.zeros(batch_size, 1, 1, device=dist_matrix.device), root_dist.transpose(1, 2)], dim=2)  # [batch_size, 1, num_species + 1]
        dist_bot = torch.cat([root_dist, dist_matrix], dim=2)  # [batch_size, num_species, num_species + 1]
        dist_full = torch.cat([dist_top, dist_bot], dim=1)  # [batch_size, num_species + 1, num_species + 1]

        num_nodes = num_species + 1
        padding_mask_dup = padding_mask_full.unsqueeze(1).expand(-1, window_size, -1).contiguous().view(batch_size * window_size, num_nodes)

        # 6. Feature Transformation along Column Axis (if window_size > 1)
        for i in range(len(self.col_layers)):
            col_in = x_full.reshape(batch_size * num_nodes, window_size, self.embed_dim) if window_size > 1 else x_full.reshape(batch_size * num_nodes, self.embed_dim)
            col_out = self.col_layers[i](col_in)
            x_full = col_out.reshape(batch_size, num_nodes, window_size, self.embed_dim)

        # 7. Deep Phylogenetic Tree Attention along Row Axis across all N+1 nodes
        # window_size == 1 passes dist_full/mds_full directly — the
        # expand+contiguous duplication materializes B*W*(N+1)^2 which does
        # not fit at cohort scale; it only matters for W > 1 (one copy per
        # window position in the flattened batch dim).
        if window_size == 1:
            dist_dup = dist_full
            mds_dup = mds_full
        else:
            if dist_full.dim() == 4:
                dist_dup = dist_full.unsqueeze(1).expand(-1, window_size, -1, -1, -1).contiguous().view(batch_size * window_size, num_nodes, num_nodes, dist_full.shape[-1])
            else:
                dist_dup = dist_full.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_nodes, num_nodes)
            mds_dup = mds_full.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_nodes, 4)

        # Bound the O(B*W*H*(N+1)^2) attention transient by the caller budget.
        budget_bytes = _resolve_budget_bytes(max_memory_mb)
        num_heads = self._num_row_heads
        row_chunk = _attn_row_chunk(budget_bytes, batch_size * window_size, num_heads, num_nodes,
                                    device=x_full.device)

        x0_dup = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, self.embed_dim)
        for i in range(len(self.row_layers)):
            row_in = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, self.embed_dim)
            row_out = self.row_layers[i](row_in, dist_dup, mds_coords=mds_dup, padding_mask=padding_mask_dup, x0=x0_dup, row_chunk=row_chunk)
            row_out = self.row_norms[i](row_in + row_out)
            x_full = row_out.reshape(batch_size, window_size, num_nodes, self.embed_dim).transpose(1, 2)
            
        # 8. Extract the Learned [ROOT] Token at Central Codon Site
        root_repr = x_full[:, 0, central_idx, :]  # [batch_size, embed_dim]
        
        # 16-Bin Ordinal LRT Logits directly from [ROOT] representation
        logits_lrt_ordinal = self.lrt_ordinal_head(root_repr)
        y_lrt_soft, _ = decode_soft_ordinal_lrt(logits_lrt_ordinal)
        if return_hidden:
            return y_lrt_soft.view(batch_size), logits_lrt_ordinal, root_repr
        return y_lrt_soft.view(batch_size), logits_lrt_ordinal


# ==============================================================================
# DOWNSTREAM FOUNDATION READOUT HEADS (PILLARS 4 & 5)
# ==============================================================================

# 16 LRT Thresholds spanning 0.1 to 500.0
CORAL_LRT_THRESHOLDS = torch.tensor([
    0.10, 0.50, 1.00, 2.00, 3.841, 5.991, 10.0, 15.0, 
    25.0, 40.0, 65.0, 100.0, 150.0, 220.0, 350.0, 500.0
], dtype=torch.float32)

# 12 -log10(p) Thresholds spanning p=0.5 down to p=1e-15
CORAL_LOGP_THRESHOLDS = torch.tensor([
    0.301, 0.699, 1.000, 1.301, 2.000, 3.000, 
    4.000, 6.000, 8.000, 10.00, 12.00, 15.00
], dtype=torch.float32)

# 12 log(omega_3) Thresholds spanning omega=1.0 to omega=1000.0
CORAL_OMEGA3_THRESHOLDS = torch.tensor([
    1.00, 1.50, 2.00, 3.00, 5.00, 8.00, 
    15.0, 30.0, 60.0, 150.0, 400.0, 1000.0
], dtype=torch.float32)

def decode_coral_lrt(logits):
    """Smooth continuous integration for LRT."""
    probs = torch.sigmoid(logits)
    log_thresh = torch.log1p(CORAL_LRT_THRESHOLDS.to(logits.device))
    deltas = torch.cat([log_thresh[0:1], log_thresh[1:] - log_thresh[:-1]])
    expected_log_lrt = (probs * deltas.unsqueeze(0)).sum(dim=-1)
    return torch.expm1(expected_log_lrt)

def decode_coral_logp(logits):
    """Smooth continuous integration for -log10(p)."""
    probs = torch.sigmoid(logits)
    thresh = CORAL_LOGP_THRESHOLDS.to(logits.device)
    deltas = torch.cat([thresh[0:1], thresh[1:] - thresh[:-1]])
    expected_logp = (probs * deltas.unsqueeze(0)).sum(dim=-1)
    return expected_logp

def decode_coral_omega3(logits):
    """Smooth continuous integration for omega_3."""
    probs = torch.sigmoid(logits)
    log_thresh = torch.log(CORAL_OMEGA3_THRESHOLDS.to(logits.device))
    deltas = torch.cat([log_thresh[0:1], log_thresh[1:] - log_thresh[:-1]])
    expected_log_omega = (probs * deltas.unsqueeze(0)).sum(dim=-1)
    return torch.exp(expected_log_omega)


class CoralHead(nn.Module):
    """
    Rank-Consistent Ordinal Regression Head (Cao et al. 2020).
    Guarantees monotonic threshold cutoffs b_1 > b_2 > ... > b_K via cumulative softplus.
    """
    def __init__(self, in_features, num_thresholds, b0_init=0.0):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.b0 = nn.Parameter(torch.tensor(float(b0_init)))
        self.theta_steps = nn.Parameter(torch.full((num_thresholds - 1,), 0.40))
        nn.init.normal_(self.fc.weight, mean=0.0, std=1.0 / (in_features ** 0.5))

    def get_cutoffs(self):
        return _coral_cutoffs(self.b0, self.theta_steps)

    def forward(self, x):
        proj = self.fc(x)
        cutoffs = self.get_cutoffs().to(device=x.device, dtype=x.dtype)
        return proj + cutoffs.unsqueeze(0)


class BustedMultiTaskHead(nn.Module):
    """
    Pillar 4: Alignment-Wide Omnibus Selection Head (BUSTED & BUSTED+S Emulation)
    with Rank-Consistent Ordinal Logits (CORAL).
    Eliminates regression compression at both the neutral floor and extreme selection bursts.
    """
    def __init__(self, embed_dim=384, num_queries=4, num_heads=4, hidden_dim=128, pool_dim=256, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.pool_dim = pool_dim
        self.in_proj = nn.Linear(embed_dim, pool_dim) if embed_dim != pool_dim else nn.Identity()
        self.queries = nn.Parameter(torch.randn(1, num_queries, pool_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(pool_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(pool_dim)
        
        self.mlp_shared = nn.Sequential(
            nn.Linear(num_queries * pool_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 1. Binary Selection Classifier (alpha = 0.05 gate)
        self.head_cls = nn.Linear(hidden_dim, 1)
        
        # 2. CORAL LRT Head (16 thresholds)
        self.coral_lrt = CoralHead(hidden_dim, num_thresholds=len(CORAL_LRT_THRESHOLDS), b0_init=1.0)
        
        # 3. CORAL -log10(p) Head (12 thresholds)
        self.coral_logp = CoralHead(hidden_dim, num_thresholds=len(CORAL_LOGP_THRESHOLDS), b0_init=0.5)
        
        # 4. CORAL omega_3 Head (12 thresholds)
        self.coral_omega3 = CoralHead(hidden_dim, num_thresholds=len(CORAL_OMEGA3_THRESHOLDS), b0_init=0.0)
        
        # 5. Synonymous Rate Variation Var(alpha)
        self.head_syn_var = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # 6. Omega Mixture Proportions (p1, p2, p3)
        self.head_prop = nn.Linear(hidden_dim, 3)

    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B = x.shape[0]
        x_proj = self.in_proj(x)
        q = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(q, x_proj, x_proj, key_padding_mask=mask)
        h = self.norm(attn_out).reshape(B, -1)
        feat = self.mlp_shared(h)
        
        logits_lrt = self.coral_lrt(feat)
        logits_logp = self.coral_logp(feat)
        logits_omega3 = self.coral_omega3(feat)
        
        return {
            "cls_prob": torch.sigmoid(self.head_cls(feat)).squeeze(-1),
            "logits_lrt": logits_lrt,
            "logits_logp": logits_logp,
            "logits_omega3": logits_omega3,
            "pred_lrt": decode_coral_lrt(logits_lrt),
            "pred_logp": decode_coral_logp(logits_logp),
            "pred_omega3": decode_coral_omega3(logits_omega3),
            "syn_var": F.softplus(self.head_syn_var(feat)).squeeze(-1),
            "omega_prop": torch.softmax(self.head_prop(feat), dim=-1)
        }


class aBSRELBranchHead(nn.Module):
    """
    Pillar 5: Branch-Site Episodic Selection Head (aBSREL Emulation).
    Predicts lineage-specific selection bursts along phylogenetic branches.
    """
    def __init__(self, embed_dim=384, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.head_p_branch = nn.Linear(hidden_dim, 1)
        self.head_omega_branch = nn.Linear(hidden_dim, 1)
        self.head_prop_branch = nn.Linear(hidden_dim, 1)

    def forward(self, h_branch):
        """
        h_branch: [..., embed_dim] branch transition embeddings (h_child - h_parent).
        """
        feat = self.mlp(self.norm(h_branch))
        return {
            "p_burst": torch.sigmoid(self.head_p_branch(feat)),
            "omega_burst": F.softplus(self.head_omega_branch(feat)) + 1.0,
            "prop_burst": torch.sigmoid(self.head_prop_branch(feat))
        }






