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


# --- Stable Attention Module with Block-Diagonal Disentanglement ---
class StableAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = BlockLinear(embed_dim, embed_dim)
        self.k_proj = BlockLinear(embed_dim, embed_dim)
        self.v_proj = BlockLinear(embed_dim, embed_dim)
        self.out_proj = BlockLinear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, q_seq_len, _ = query.shape
        k_seq_len = key.shape[1]
        
        q_proj = self.q_proj(query)
        k_proj = self.k_proj(key)
        v_proj = self.v_proj(value)
        
        q_h = q_proj.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_h = k_proj.view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = v_proj.view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e4)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = torch.where(mask, torch.zeros_like(attn_weights), attn_weights)
        else:
            attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v_h)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.embed_dim)
        return self.out_proj(out)


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
        
        # 3-Channel Unrooted Tree Topological Attention Projection:
        # Channel 0: Patristic Path Distance D_ij
        # Channel 1: Topological Node Count N_ij
        # Channel 2: Off-Path Subtree Density S_ij
        self.tree_w1 = nn.Parameter(torch.randn(num_heads, 3) * 0.02)
        self.tree_b1 = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.tree_w2 = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.02)
        
        # Legacy fallback support for 1D distance inputs
        self.phylo_w1 = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.02)
        self.phylo_b1 = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.phylo_w2 = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.02)
        
        # Dynamic Site-Level Tree Rate Scaler (MEME alpha_s site rate scaler intuition)
        self.site_tree_scaler = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.site_tree_scaler.weight)
        nn.init.zeros_(self.site_tree_scaler.bias)
        
        half_dim = self.head_dim // 2
        self.rope_freqs = nn.Parameter(torch.randn(num_heads, half_dim, 4) * 0.05)
        # Learnable Initial-Representation Skip Weight (Initialized to 0.20)
        self.alpha_skip = nn.Parameter(torch.tensor(0.20))
        
        self.out_proj = BlockLinear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, dist_matrix, mds_coords=None, padding_mask=None, nonsyn_mask=None, syn_mask=None, x0=None):
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


class StableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = StableAttention(d_model, nhead, dropout)
        
        self.linear1 = BlockLinear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = BlockLinear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src):
        attn_out = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout1(attn_out))
        
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))
        return src


# --- Fitch Codon Parsimony and Tree Topology Utilities ---
CODON_TO_AA_DICT = {
    'TTT': 0, 'TTC': 0, 'TTA': 1, 'TTG': 1, 'TCT': 2, 'TCC': 2, 'TCA': 2, 'TCG': 2,
    'TAT': 3, 'TAC': 3, 'TAA': 20, 'TAG': 20, 'TGT': 4, 'TGC': 4, 'TGA': 20, 'TGG': 5,
    'CTT': 1, 'CTC': 1, 'CTA': 1, 'CTG': 1, 'CCT': 6, 'CCC': 6, 'CCA': 6, 'CCG': 6,
    'CAT': 7, 'CAC': 7, 'CAA': 8, 'CAG': 8, 'CGT': 9, 'CGC': 9, 'CGA': 9, 'CGG': 9,
    'ATT': 10, 'ATC': 10, 'ATA': 10, 'ATG': 11, 'ACT': 12, 'ACC': 12, 'ACA': 12, 'ACG': 12,
    'AAT': 13, 'AAC': 13, 'AAA': 14, 'AAG': 14, 'AGT': 2, 'AGC': 2, 'AGA': 9, 'AGG': 9,
    'GTT': 15, 'GTC': 15, 'GTA': 15, 'GTG': 15, 'GCT': 16, 'GCC': 16, 'GCA': 16, 'GCG': 16,
    'GAT': 17, 'GAC': 17, 'GAA': 18, 'GAG': 18, 'GGT': 19, 'GGC': 19, 'GGA': 19, 'GGG': 19
}

NUC_LIST = ['T', 'C', 'A', 'G']
SENSE_CODONS = [n1+n2+n3 for n1 in NUC_LIST for n2 in NUC_LIST for n3 in NUC_LIST if CODON_TO_AA_DICT[n1+n2+n3] < 20]
SENSE_CODON_TO_IDX = {c: i for i, c in enumerate(SENSE_CODONS)}
SERINE_TCT_SET = {SENSE_CODON_TO_IDX[c] for c in ['TCT', 'TCC', 'TCA', 'TCG']}
SERINE_AGC_SET = {SENSE_CODON_TO_IDX[c] for c in ['AGT', 'AGC']}

def build_tree_topology(newick_str, selected_species):
    clean_newick = newick_str.split(";")[0].strip() + ";" if newick_str else ""
    if not clean_newick:
        N = len(selected_species)
        num_nodes = 2 * N - 1
        parent_array = np.full(num_nodes, -1, dtype=np.int32)
        for i in range(N):
            parent_array[i] = N + (i // 2) if (N + (i // 2)) < num_nodes else num_nodes - 1
        branch_lengths = np.ones(num_nodes, dtype=np.float32) * 0.05
        return parent_array, branch_lengths
        
    try:
        root = parse_newick(clean_newick)
    except Exception:
        N = len(selected_species)
        num_nodes = 2 * N - 1
        parent_array = np.full(num_nodes, -1, dtype=np.int32)
        for i in range(N):
            parent_array[i] = N + (i // 2) if (N + (i // 2)) < num_nodes else num_nodes - 1
        branch_lengths = np.ones(num_nodes, dtype=np.float32) * 0.05
        return parent_array, branch_lengths
        
    species_to_idx = {}
    norm_selected = [s.replace("'", "").replace('"', '').strip() for s in selected_species]
    for idx, s in enumerate(norm_selected):
        species_to_idx[s] = idx
        
    N = len(selected_species)
    num_nodes = 2 * N - 1
    node_to_id = {}
    
    all_nodes = []
    stack = [root]
    while stack:
        curr = stack.pop()
        all_nodes.append(curr)
        for c in reversed(curr.children):
            stack.append(c)
            
    leaves = [n for n in all_nodes if not n.children]
    internals = [n for n in all_nodes if n.children]
    
    leaves_found = 0
    for term in leaves:
        term_name = term.name.replace("'", "").replace('"', '').strip() if term.name else ""
        if term_name in species_to_idx:
            idx = species_to_idx[term_name]
            node_to_id[term] = idx
            leaves_found += 1
            
    if leaves_found < N:
        for idx, term in enumerate(leaves):
            if idx < N and term not in node_to_id:
                node_to_id[term] = idx
                
    next_int_id = N
    for n in internals:
        node_to_id[n] = next_int_id
        next_int_id += 1
        if next_int_id >= num_nodes:
            break
            
    parent_array = np.full(num_nodes, -1, dtype=np.int32)
    branch_lengths = np.ones(num_nodes, dtype=np.float32) * 1e-3
    
    for n, n_id in node_to_id.items():
        branch_lengths[n_id] = max(float(n.length), 1e-4)
        if n.parent and n.parent in node_to_id:
            parent_array[n_id] = node_to_id[n.parent]
            
    return parent_array, branch_lengths

def build_sankoff_cost_matrix():
    cost_matrix = np.zeros((61, 61), dtype=np.float32)
    for i, c1 in enumerate(SENSE_CODONS):
        aa1 = CODON_TO_AA_DICT[c1]
        for j, c2 in enumerate(SENSE_CODONS):
            if i == j:
                cost_matrix[i, j] = 0.0
                continue
            aa2 = CODON_TO_AA_DICT[c2]
            nuc_diff = sum(1 for k in range(3) if c1[k] != c2[k])
            
            if aa1 == aa2:
                cost_matrix[i, j] = 1.0 * nuc_diff
            else:
                cost_matrix[i, j] = 2.5 * nuc_diff
    return cost_matrix

SANKOFF_COST_MATRIX = build_sankoff_cost_matrix()

def fitch_codon_parsimony(site_codon_ids, parent_array, branch_lengths, max_k=32):
    num_nodes = len(parent_array)
    num_taxa = (num_nodes + 1) // 2
    
    S = np.zeros((num_nodes, 61), dtype=np.float32)
    
    for i in range(min(num_taxa, len(site_codon_ids))):
        c_tok = site_codon_ids[i]
        if c_tok < 64:
            c_str = codons_list[c_tok] if c_tok < 64 else '???'
            s_idx = SENSE_CODON_TO_IDX.get(c_str, 61)
            if s_idx < 61:
                S[i, :] = 1e6
                S[i, s_idx] = 0.0
            else:
                S[i, :] = 0.0
        else:
            S[i, :] = 0.0
            
    children = [[] for _ in range(num_nodes)]
    for v in range(num_nodes):
        p = parent_array[v]
        if p >= 0:
            children[p].append(v)
            
    root = -1
    for v in range(num_nodes):
        if parent_array[v] < 0 and len(children[v]) > 0:
            root = v
            break
    if root < 0:
        root = num_nodes - 1
        
    # Dynamic Post-Order Bottom-Up DP (children before parent)
    post_order = []
    def get_post_order(u):
        for ch in children[u]:
            get_post_order(ch)
        post_order.append(u)
    get_post_order(root)
    
    for u in post_order:
        ch = children[u]
        if len(ch) > 0:
            node_cost = np.zeros(61, dtype=np.float32)
            for child in ch:
                ch_cost_matrix = S[child, :][np.newaxis, :] + SANKOFF_COST_MATRIX
                node_cost += np.min(ch_cost_matrix, axis=1)
            S[u, :] = node_cost
            
    # Dynamic Pre-Order Top-Down Backtracking (parent before children)
    pre_order = post_order[::-1]
    reconstructed = np.zeros(num_nodes, dtype=np.int32)
    reconstructed[root] = np.argmin(S[root, :])
    
    for u in pre_order[1:]:
        p = parent_array[u]
        p_state = reconstructed[p]
        costs = S[u, :] + SANKOFF_COST_MATRIX[p_state, :]
        reconstructed[u] = np.argmin(costs)

        
    active_edges = []
    total_syn_count = 0.0
    total_nonsyn_count = 0.0
    
    for v in range(num_nodes - 1):
        p = parent_array[v]
        if p < 0:
            continue
        c_u = reconstructed[p]
        c_v = reconstructed[v]
        
        if c_u != c_v and c_u < 61 and c_v < 61:
            b_len = max(float(branch_lengths[v]), 1e-4)
            sub_id = c_u * 61 + c_v
            
            aa_u = CODON_TO_AA_DICT[SENSE_CODONS[c_u]]
            aa_v = CODON_TO_AA_DICT[SENSE_CODONS[c_v]]
            
            str_u, str_v = SENSE_CODONS[c_u], SENSE_CODONS[c_v]
            nuc_diff = sum(1 for i in range(3) if str_u[i] != str_v[i])
            
            is_syn = 1.0 if aa_u == aa_v else 0.0
            is_nonsyn_single = 1.0 if (aa_u != aa_v and nuc_diff == 1) else 0.0
            is_nonsyn_multi = 1.0 if (aa_u != aa_v and nuc_diff > 1) else 0.0
            is_serine = 1.0 if (aa_u == aa_v and ((c_u in SERINE_TCT_SET and c_v in SERINE_AGC_SET) or (c_u in SERINE_AGC_SET and c_v in SERINE_TCT_SET))) else 0.0
            
            if is_syn == 1.0:
                total_syn_count += 1.0
            else:
                total_nonsyn_count += 1.0
                
            rate = 1.0 / b_len
            active_edges.append((is_syn, rate, sub_id, [is_syn, is_nonsyn_single, is_nonsyn_multi, is_serine], b_len))
            
    dNdS_ratio = total_nonsyn_count / (total_syn_count + 0.1)
    rates = [e[1] for e in active_edges]
    mean_rate = float(np.mean(rates)) if len(rates) > 0 else 1.0
    
    nonsyn_edges = [e for e in active_edges if e[0] == 0.0]
    syn_edges = [e for e in active_edges if e[0] == 1.0]
    
    nonsyn_edges.sort(key=lambda x: x[1], reverse=True)
    syn_edges.sort(key=lambda x: x[1], reverse=True)
    
    # Dynamic dual allocation ratio: 75% non-synonymous, 25% synonymous
    target_nonsyn = int(max_k * 0.75)
    target_syn = max_k - target_nonsyn
    
    k_nonsyn = min(target_nonsyn, len(nonsyn_edges))
    k_syn = min(target_syn, len(syn_edges))
    
    selected = nonsyn_edges[:k_nonsyn] + syn_edges[:k_syn]
    rem = nonsyn_edges[k_nonsyn:] + syn_edges[k_syn:]
    rem.sort(key=lambda x: x[1], reverse=True)
    
    if len(selected) < max_k:
        selected += rem[:(max_k - len(selected))]
        
    sub_ids = np.zeros(max_k, dtype=np.int64)
    flags = np.zeros((max_k, 8), dtype=np.float32)
    lengths = np.ones(max_k, dtype=np.float32) * 1e-4
    mask = np.zeros(max_k, dtype=np.float32)
    
    for i, (_, rate, sub_id, fl, b_len) in enumerate(selected):
        sub_ids[i] = sub_id
        # Pure Transformer: Zero out all precomputed summary heuristics (dNdS_ratio, total_nonsyn, total_syn, burst_ratio)
        flags[i] = fl + [0.0, 0.0, 0.0, 0.0]
        lengths[i] = b_len
        mask[i] = 1.0
        
    return sub_ids, flags, lengths, mask


def _build_path_ns_tensor():
    # 61x61x2 precomputed lookup table of expected (N, S) steps
    sense_codons = ['AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT', 'GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAC', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT']
    code = {'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R', 'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 'TAC':'Y', 'TAT':'Y', 'TGC':'C', 'TGT':'C', 'TGG':'W'}
    stops = {'TAA', 'TAG', 'TGA'}
    
    import itertools
    matrix = np.zeros((61, 61, 2), dtype=np.float32)
    for i, c1 in enumerate(sense_codons):
        for j, c2 in enumerate(sense_codons):
            if c1 == c2:
                continue
            diffs = [k for k in range(3) if c1[k] != c2[k]]
            perms = list(itertools.permutations(diffs))
            valid_paths = []
            for perm in perms:
                path = [c1]
                curr = list(c1)
                valid = True
                for pos in perm:
                    curr[pos] = c2[pos]
                    nc = "".join(curr)
                    if nc in stops:
                        valid = False
                        break
                    path.append(nc)
                if valid:
                    valid_paths.append(path)
            if not valid_paths:
                for perm in perms:
                    path = [c1]
                    curr = list(c1)
                    for pos in perm:
                        curr[pos] = c2[pos]
                        path.append("".join(curr))
                    valid_paths.append(path)
            tn, ts = 0.0, 0.0
            for p in valid_paths:
                pn, ps = 0, 0
                for step in range(len(p) - 1):
                    if code.get(p[step]) == code.get(p[step+1]):
                        ps += 1
                    else:
                        pn += 1
                tn += pn
                ts += ps
            matrix[i, j, 0] = tn / len(valid_paths)
            matrix[i, j, 1] = ts / len(valid_paths)
    return torch.tensor(matrix, dtype=torch.float32)


# --- Sparse Codon Edge Token Encoder ---
class SparseCodonEdgeEncoder(nn.Module):
    def __init__(self, embed_dim=128, max_k=64, num_categories=8):
        super().__init__()
        self.max_k = max_k
        self.embed_dim = embed_dim
        
        self.register_buffer('path_ns_matrix', _build_path_ns_tensor())
        self.codon_sub_embed = nn.Embedding(3721, 64)
        self.category_proj = nn.Linear(num_categories, 32)
        
        self.b_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 32)
        )
        
        # Additional path projection layer for (exp_N, exp_S, nonsyn_ratio, nonsyn_flux, syn_flux, diff_flux)
        self.path_proj = nn.Sequential(
            nn.Linear(6, 16),
            nn.GELU(),
            nn.Linear(16, 16)
        )
        
        self.edge_proj = nn.Sequential(
            nn.Linear(64 + 32 + 32 + 16, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.pool_combine = nn.Sequential(
            nn.Linear(5 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, active_sub_ids, active_flags, active_lengths, active_mask):
        if self.category_proj.in_features == 4 and active_flags.shape[-1] >= 4:
            cat_flags = active_flags[..., :4]
        elif self.category_proj.in_features == 7 and active_flags.shape[-1] >= 7:
            cat_flags = active_flags[..., :7]
        elif self.category_proj.in_features == 8 and active_flags.shape[-1] < 8:
            pad_size = 8 - active_flags.shape[-1]
            pad_tensor = torch.zeros((*active_flags.shape[:-1], pad_size), device=active_flags.device, dtype=active_flags.dtype)
            cat_flags = torch.cat([active_flags, pad_tensor], dim=-1)
        else:
            cat_flags = active_flags
            
        sub_emb = self.codon_sub_embed(active_sub_ids)
        cat_emb = self.category_proj(cat_flags)
        
        log_b = torch.log(torch.clamp(active_lengths, min=1e-4))
        b_feat = torch.stack([active_lengths, log_b], dim=-1)
        b_emb = self.b_mlp(b_feat)
        
        # Extract path-averaged (N, S) metrics from lookup matrix
        c_u = torch.clamp(active_sub_ids // 61, min=0, max=60)
        c_v = torch.clamp(active_sub_ids % 61, min=0, max=60)
        ns_vals = self.path_ns_matrix[c_u, c_v] # [B, K, 2]
        exp_N = ns_vals[..., 0] # [B, K]
        exp_S = ns_vals[..., 1] # [B, K]
        
        b_len_clamp = torch.clamp(active_lengths, min=1e-4)
        
        # Solution 1: Composition-Aware Path-Averaged Features
        nonsyn_ratio = exp_N / (exp_N + exp_S + 1e-6)
        nonsyn_flux = exp_N / b_len_clamp
        syn_flux = exp_S / b_len_clamp
        diff_flux = (exp_N - exp_S) / b_len_clamp
        
        path_feats = torch.stack([exp_N, exp_S, nonsyn_ratio, nonsyn_flux, syn_flux, diff_flux], dim=-1)
        path_emb = self.path_proj(path_feats)
        
        concat_feat = torch.cat([sub_emb, cat_emb, b_emb, path_emb], dim=-1)
        edge_vec = self.edge_proj(concat_feat)
        
        intensity = 1.0 / (torch.clamp(active_lengths, min=1e-4) + 1e-3)
        intensity_log = torch.log1p(torch.clamp(intensity, max=100.0))
        
        weighted_edge_vec = edge_vec * intensity_log.unsqueeze(-1) * active_mask.unsqueeze(-1)
        
        # 1. Max Pooling across active edges (isolates 1st highest burst)
        masked_for_max = weighted_edge_vec.masked_fill((active_mask == 0).unsqueeze(-1), -1e4)
        max_pooled = torch.relu(torch.max(masked_for_max, dim=1)[0])
        
        # 2. Mean Pooling across active edges (tree-size invariant rate average)
        active_counts = active_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pooled = weighted_edge_vec.sum(dim=1) / active_counts
        
        # 3. L2 Norm Pooling (overall mutational energy normalized by active counts)
        l2_pooled = torch.sqrt((weighted_edge_vec ** 2).sum(dim=1) / active_counts + 1e-6)
        
        # 4. Softmax Attention Pooling (weighted by edge intensity)
        attn_logits = (weighted_edge_vec.sum(dim=-1) / math.sqrt(self.embed_dim)).masked_fill(active_mask == 0, -1e4)
        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)
        attn_pooled = (weighted_edge_vec * attn_weights).sum(dim=1)
        
        # 5. Top-2 Edge Pooling (isolates 2nd highest burst, zeroed if < 2 active edges)
        has_at_least_two_edges = (active_counts >= 2.0).float()
        top2_val, _ = torch.topk(masked_for_max, k=min(2, masked_for_max.shape[1]), dim=1)
        if top2_val.shape[1] >= 2:
            top2_pooled = torch.relu(top2_val[:, 1, :]) * has_at_least_two_edges
        else:
            top2_pooled = max_pooled * has_at_least_two_edges
            
        has_edges = (active_counts > 0.0).float()
        combined = torch.cat([max_pooled, mean_pooled, l2_pooled, attn_pooled, top2_pooled], dim=-1)
        out_vec = F.layer_norm(self.pool_combine(combined), (self.embed_dim,))
        return torch.where(has_edges > 0, out_vec, torch.zeros_like(out_vec))



# --- 17-Bin Ordinal Likelihood Partition & Soft-Bin Expectation Decoder ---
BIN_EDGES_9 = [0.0, 0.2738, 1.8272, 3.1248, 4.4537, 5.7987, 7.5909, 12.1310, 16.6963, 21.2737, 100.0]
BIN_EDGES_12 = [0.0, 0.2738, 0.7500, 1.2500, 1.8272, 2.4500, 3.1248, 4.4537, 5.7987, 7.5909, 12.1310, 16.6963, 21.2737, 100.0]
BIN_EDGES_16 = [0.0, 0.2738, 0.7500, 1.2500, 1.8272, 2.4500, 3.1248, 4.4537, 5.7987, 7.5909, 12.1310, 16.6963, 22.0, 32.0, 50.0, 75.0, 100.0]
BIN_EDGES = BIN_EDGES_16
BIN_MEANS = torch.tensor([0.00, 0.51, 1.00, 1.54, 2.14, 2.79, 3.79, 5.13, 6.69, 9.86, 14.41, 18.98, 35.00])

def sparsemax(logits, dim=-1):
    """
    TPU-friendly Sparsemax (Martins & Astudillo, ICML 2016).
    Projects logits onto the probability simplex, truncating low-scoring tail values to EXACTLY 0.0.
    Uses 100% static tensor shapes and ops to prevent PyTorch-XLA recompilation graph breaks.
    """
    input_sorted, _ = torch.sort(logits, descending=True, dim=dim)
    cumsum = torch.cumsum(input_sorted, dim=dim)
    
    num_elements = logits.shape[dim]
    k_range = torch.arange(1, num_elements + 1, device=logits.device, dtype=logits.dtype)
    shape = [1] * logits.dim()
    shape[dim] = -1
    k_range = k_range.view(*shape)
    
    bound = 1.0 + k_range * input_sorted
    is_greater = (bound > cumsum).float()
    
    k_max = torch.max(is_greater * k_range, dim=dim, keepdim=True)[0]
    tau = (torch.gather(cumsum, dim, k_max.long() - 1) - 1.0) / k_max
    
    return torch.relu(logits - tau)


def decode_soft_ordinal_lrt(logits_ordinal, bin_edges=None, temperature=1.0):
    """
    Rigorously decodes continuous LRT prediction from CORAL cumulative ordinal logits
    using the Cumulative Survival Function Integral Theorem: E[Y] = int_0^inf P(Y > y) dy.
    """
    num_heads = logits_ordinal.shape[-1] if logits_ordinal.dim() > 1 else (logits_ordinal.shape[0] if logits_ordinal.dim() == 1 else 16)
    if bin_edges is None:
        if num_heads >= 15:
            bin_edges = BIN_EDGES_16
        elif num_heads >= 11:
            bin_edges = BIN_EDGES_12
        else:
            bin_edges = BIN_EDGES_9
        
    device = logits_ordinal.device
    edges = torch.tensor(bin_edges[:num_heads+1], device=device, dtype=logits_ordinal.dtype)
    widths = (edges[1:] - edges[:-1]).to(device=device, dtype=logits_ordinal.dtype)
    
    # Cumulative probabilities P(LRT > threshold_k) for k in 0..num_heads-1
    p_cum = torch.sigmoid(logits_ordinal / temperature)
    
    # E[LRT] = sum_k P(LRT > t_k) * delta_t_k
    widths_view = widths.view(*([1] * (p_cum.dim() - 1)), -1)
    y_continuous_lrt = torch.sum(p_cum * widths_view, dim=-1)
    return y_continuous_lrt, p_cum


# Fixed Empirical Prior Cutoffs b_k = logit(P(Y > T_k))
EMPIRICAL_PRIOR_CUTOFFS = torch.tensor([
    -1.7346, -2.1972, -2.5867, -2.9444, -3.3168, -3.6636,
    -4.1846, -4.5951, -5.1100, -5.8061, -6.5008, -7.1301,
    -7.8236, -8.5170, -9.2102, -9.9034
])


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


LOG_CORAL_THRESHOLDS_8 = torch.tensor([0.0000, 0.6931, 1.4170, 1.6963, 2.0327, 2.4704, 3.0445, 3.9318, 4.6151])
LOG_CORAL_DELTAS_8 = torch.tensor([0.6931, 0.7239, 0.2793, 0.3364, 0.4377, 0.5741, 0.8873, 0.6833])

LOG_CORAL_THRESHOLDS_16 = torch.tensor([0.0000, 0.2420, 0.5596, 0.8109, 1.0393, 1.2384, 1.4170, 1.6963, 1.9168, 2.1507, 2.5750, 2.8734, 3.1355, 3.4965, 3.9318, 4.3307, 4.6151])
LOG_CORAL_DELTAS_16 = torch.tensor([0.2420, 0.3176, 0.2513, 0.2284, 0.1991, 0.1786, 0.2793, 0.2205, 0.2339, 0.4243, 0.2984, 0.2621, 0.3610, 0.4353, 0.3989, 0.2844])

LOG_CORAL_THRESHOLDS_24 = torch.tensor([
    0.0000, 0.2420, 0.4055, 0.5596, 0.6931, 0.8544, 1.0393, 1.2384, 1.4170,
    1.5772, 1.6963, 1.8582, 2.0327, 2.2246, 2.4704, 2.7081, 2.9444, 3.2189,
    3.4965, 3.7612, 4.0073, 4.2341, 4.4188, 4.6151, 4.7958
])
LOG_CORAL_DELTAS_24 = torch.tensor([
    0.2420, 0.1635, 0.1542, 0.1335, 0.1613, 0.1849, 0.1991, 0.1786, 0.1602,
    0.1191, 0.1619, 0.1746, 0.1919, 0.2458, 0.2376, 0.2364, 0.2744, 0.2776,
    0.2647, 0.2461, 0.2268, 0.1847, 0.1963, 0.1807
])

# Backward compatibility aliases
LOG_CORAL_THRESHOLDS_FULL = LOG_CORAL_THRESHOLDS_16
LOG_CORAL_DELTAS_TENSOR = LOG_CORAL_DELTAS_16
LOG_CORAL_DELTAS_12 = LOG_CORAL_DELTAS_16[:12]
CORAL_THRESHOLDS_FULL = LOG_CORAL_THRESHOLDS_16
CORAL_DELTAS_TENSOR = LOG_CORAL_DELTAS_16
PURE_CORAL_DELTAS_12 = LOG_CORAL_DELTAS_12

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
        self.max_species = max_species
        self.num_layers = num_layers
        self.use_aa_embeddings = True
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
        self.col_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_col_layers)])

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
        
        # Pre-compute genetic code pairwise attention masks at central site
        c_cent = msa_codons[:, :, central_idx]  # [batch_size, num_species]
        a_cent = msa_aas[:, :, central_idx]     # [batch_size, num_species]
        
        valid = (c_cent < 64) & (a_cent < 21) & (~padding_mask.bool())
            
        v_float = valid.float()
        v_pair = v_float.unsqueeze(1) * v_float.unsqueeze(2)  # [batch_size, num_species, num_species]
        
        # Static matrix identity mask (Zero PyTorch-XLA recompilation)
        diag_mask = torch.eye(num_species, device=c_cent.device, dtype=v_float.dtype).unsqueeze(0)
        pair_mask = v_pair * (1.0 - diag_mask)
        
        a_diff = (a_cent.unsqueeze(1) != a_cent.unsqueeze(2)).float() * pair_mask
        c_diff = (c_cent.unsqueeze(1) != c_cent.unsqueeze(2)).float()
        a_eq = (a_cent.unsqueeze(1) == a_cent.unsqueeze(2)).float()
        c_syn = (c_diff * a_eq) * pair_mask
        
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
        
        # 5. Augment Non-Syn and Syn masks with zero borders for root
        nonsyn_top = torch.zeros(batch_size, 1, num_species + 1, device=a_diff.device)
        nonsyn_bot = torch.cat([torch.zeros(batch_size, num_species, 1, device=a_diff.device), a_diff], dim=2)
        nonsyn_full = torch.cat([nonsyn_top, nonsyn_bot], dim=1)
        
        syn_top = torch.zeros(batch_size, 1, num_species + 1, device=c_syn.device)
        syn_bot = torch.cat([torch.zeros(batch_size, num_species, 1, device=c_syn.device), c_syn], dim=2)
        syn_full = torch.cat([syn_top, syn_bot], dim=1)
        
        num_nodes = num_species + 1
        padding_mask_dup = padding_mask_full.unsqueeze(1).expand(-1, window_size, -1).contiguous().view(batch_size * window_size, num_nodes)
        nonsyn_mask_dup = nonsyn_full.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_nodes, num_nodes)
        syn_mask_dup = syn_full.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_nodes, num_nodes)

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
            row_out = self.row_layers[i](row_in, dist_dup, mds_coords=mds_dup, padding_mask=padding_mask_dup, nonsyn_mask=nonsyn_mask_dup, syn_mask=syn_mask_dup, x0=x0_dup)
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




