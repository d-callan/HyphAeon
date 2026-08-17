"""
axomeme/model.py
----------------
Core Neural Architecture: PhyloAxialTransformer with Multi-Scale 4D Tree-RoPE
for ultra-fast episodic positive selection inference.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # because they are present in the pretrained checkpoint (axomeme_v1.pt)
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
        
    def forward(self, x, dist_matrix, mds_coords=None, padding_mask=None, x0=None):
        batch_size, num_species, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Tree-RoPE: Apply 4D MDS Rotary Position Phase Rotations to Query & Key
        if mds_coords is not None:
            half_dim = self.head_dim // 2
            m_exp = mds_coords.unsqueeze(1).unsqueeze(3) # [batch_size, 1, num_species, 1, 4]
            f_exp = self.rope_freqs.unsqueeze(0).unsqueeze(2) # [1, num_heads, 1, half_dim, 4]
            angles = (m_exp * f_exp).sum(dim=-1) # [batch_size, num_heads, num_species, half_dim]
            cos = torch.cos(angles).to(q.dtype)
            sin = torch.sin(angles).to(q.dtype)
            
            q1, q2 = q[..., :half_dim], q[..., half_dim:]
            k1, k2 = k[..., :half_dim], k[..., half_dim:]
            
            q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
            
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Sequence Density Invariant Softmax Normalization
        if padding_mask is not None:
            active_counts = (~padding_mask).sum(dim=-1, keepdim=True).clamp(min=1.0).float()
            density_scale = torch.log(active_counts / 256.0).unsqueeze(-1).unsqueeze(-1)
            scores = scores + density_scale
        
        # Pure Continuous-Time Markov Transition Probability Tree Kernel
        # P_ij(d) = eps0 + (1 - eps0) * exp(-lambda_h * d_ij)
        # Reflects exact continuous-time Markov substitution process where transition probability asymptotes to eps0 (1/20) as d -> inf.
        dist_1d = dist_matrix[..., 0] if dist_matrix.dim() == 4 or (dist_matrix.dim() == 3 and dist_matrix.shape[-1] == 3) else dist_matrix
        bias_1d = dist_1d.unsqueeze(1) if dist_1d.dim() == 3 else dist_1d.unsqueeze(0).unsqueeze(1)
        decay_rate = F.softplus(self.phylo_w1)  # [num_heads, 1, 1] learnable rate per attention head
        eps0 = 0.05  # Stationary background frequency floor (1/20 amino acids)
        markov_kernel = eps0 + (1.0 - eps0) * torch.exp(-decay_rate * bias_1d)
        
        # Pure log-space Markov transition probability kernel (Zero unphysical linear subtraction)
        scores = scores + torch.log(markov_kernel.clamp(min=1e-5))
            
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e4)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = torch.where(mask, torch.zeros_like(attn_weights), attn_weights)
        else:
            attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights).to(v.dtype)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_species, self.embed_dim)
        out = self.out_proj(out)
        if x0 is not None:
            out = out + self.alpha_skip * x0  # Learnable initial-representation skip connection
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
        steps = F.softplus(self.theta_steps)
        cum_steps = torch.cumsum(steps, dim=0)
        cutoffs = torch.cat([self.b0.unsqueeze(0), self.b0 - cum_steps])
        return cutoffs

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

    def forward(self, msa_codons, msa_aas, dist_matrix, mds_coords, padding_mask=None):
        batch_size, num_species, window_size = msa_codons.shape
        central_idx = window_size // 2
        
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
        if dist_full.dim() == 4:
            dist_dup = dist_full.unsqueeze(1).expand(-1, window_size, -1, -1, -1).contiguous().view(batch_size * window_size, num_nodes, num_nodes, dist_full.shape[-1])
        else:
            dist_dup = dist_full.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_nodes, num_nodes)
        
        mds_dup = mds_full.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_nodes, 4)

        x0_dup = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, self.embed_dim)
        for i in range(len(self.row_layers)):
            row_in = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, self.embed_dim)
            row_out = self.row_layers[i](row_in, dist_dup, mds_coords=mds_dup, padding_mask=padding_mask_dup, x0=x0_dup)
            row_out = self.row_norms[i](row_in + row_out)
            x_full = row_out.reshape(batch_size, window_size, num_nodes, self.embed_dim).transpose(1, 2)
            
        # 8. Extract the Learned [ROOT] Token at Central Codon Site
        root_repr = x_full[:, 0, central_idx, :]  # [batch_size, embed_dim]
        
        # 16-Bin Ordinal LRT Logits directly from [ROOT] representation
        logits_lrt_ordinal = self.lrt_ordinal_head(root_repr)
        
        if self.training:
            return logits_lrt_ordinal
        else:
            y_lrt_soft, _ = decode_soft_ordinal_lrt(logits_lrt_ordinal)
            return y_lrt_soft.view(batch_size), logits_lrt_ordinal




