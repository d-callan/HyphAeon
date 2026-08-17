"""
axomeme/dataset.py
------------------
Data preprocessing, tokenization, tree patristic distance calculation,
classical 4D MDS embedding, and alignment parsing.
"""

import numpy as np
import scipy.stats as stats
import torch
from Bio import SeqIO, Phylo

GENETIC_CODE = {
    'TTT': 0, 'TTC': 1, 'TTA': 2, 'TTG': 3, 'TCT': 4, 'TCC': 5, 'TCA': 6, 'TCG': 7,
    'TAT': 8, 'TAC': 9, 'TAA': 64, 'TAG': 64, 'TGT': 10, 'TGC': 11, 'TGA': 64, 'TGG': 12,
    'CTT': 13, 'CTC': 14, 'CTA': 15, 'CTG': 16, 'CCT': 17, 'CCC': 18, 'CCA': 19, 'CCG': 20,
    'CAT': 21, 'CAC': 22, 'CAA': 23, 'CAG': 24, 'CGT': 25, 'CGC': 26, 'CGA': 27, 'CGG': 28,
    'ATT': 29, 'ATC': 30, 'ATA': 31, 'ATG': 32, 'ACT': 33, 'ACC': 34, 'ACA': 35, 'ACG': 36,
    'AAT': 37, 'AAC': 38, 'AAA': 39, 'AAG': 40, 'AGT': 41, 'AGC': 42, 'AGA': 43, 'AGG': 44,
    'GTT': 45, 'GTC': 46, 'GTA': 47, 'GTG': 48, 'GCT': 49, 'GCC': 50, 'GCA': 51, 'GCG': 52,
    'GAT': 53, 'GAC': 54, 'GAA': 55, 'GAG': 56, 'GGT': 57, 'GGC': 58, 'GGA': 59, 'GGG': 60
}

AA_MAP = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

CODON_TO_AA = {
    'TTT':'F', 'TTC':'F', 'TTA':'L', 'TTG':'L', 'TCT':'S', 'TCC':'S', 'TCA':'S', 'TCG':'S',
    'TAT':'Y', 'TAC':'Y', 'TAA':'*', 'TAG':'*', 'TGT':'C', 'TGC':'C', 'TGA':'*', 'TGG':'W',
    'CTT':'L', 'CTC':'L', 'CTA':'L', 'CTG':'L', 'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P',
    'CAT':'H', 'CAC':'H', 'CAA':'Q', 'CAG':'Q', 'CGT':'R', 'CGC':'R', 'CGA':'R', 'CGG':'R',
    'ATT':'I', 'ATC':'I', 'ATA':'I', 'ATG':'M', 'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T',
    'AAT':'N', 'AAC':'N', 'AAA':'K', 'AAG':'K', 'AGT':'S', 'AGC':'S', 'AGA':'R', 'AGG':'R',
    'GTT':'V', 'GTC':'V', 'GTA':'V', 'GTG':'V', 'GCT':'A', 'GCC':'A', 'GCA':'A', 'GCG':'A',
    'GAT':'D', 'GAC':'D', 'GAA':'E', 'GAG':'E', 'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G'
}

def get_codon_token(codon: str) -> int:
    return GENETIC_CODE.get(codon.upper(), 64)

def get_aa_token(codon: str) -> int:
    aa = CODON_TO_AA.get(codon.upper(), '-')
    return AA_MAP.get(aa, 20)

def compute_fast_dist_matrix(tree, taxa):
    n = len(taxa)
    dist_mat = np.zeros((n, n), dtype=np.float32)
    taxa_set = set(taxa)
    terminals = {t.name: t for t in tree.get_terminals() if t.name in taxa_set}
    
    root = tree.root
    depths = {}
    def calc_depths(node, current_depth=0.0):
        depths[id(node)] = current_depth
        for child in node.clades:
            bl = child.branch_length if child.branch_length is not None else 0.0
            calc_depths(child, current_depth + bl)
    calc_depths(root)
    
    parents = {}
    def get_parents(node):
        for child in node.clades:
            parents[id(child)] = node
            get_parents(child)
    get_parents(root)
    
    for i in range(n):
        ti_name = taxa[i]
        ti = terminals.get(ti_name)
        if ti is None: continue
        
        path_i = [id(ti)]
        curr = id(ti)
        while curr in parents:
            curr = id(parents[curr])
            path_i.append(curr)
        path_i_set = set(path_i)
        
        for j in range(i, n):
            tj_name = taxa[j]
            tj = terminals.get(tj_name)
            if tj is None: continue
            
            curr = id(tj)
            while curr not in path_i_set:
                curr = id(parents[curr])
            lca_id = curr
            d = depths[id(ti)] + depths[id(tj)] - 2.0 * depths[lca_id]
            dist_mat[i, j] = d
            dist_mat[j, i] = d
            
    return dist_mat

def compute_mds_coordinates(dist_matrix: np.ndarray, n_components: int = 4) -> np.ndarray:
    n = dist_matrix.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H.dot(dist_matrix ** 2).dot(H)
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    pos_eigvals = np.maximum(eigvals[:n_components], 0)
    coords = eigvecs[:, :n_components] * np.sqrt(pos_eigvals)
    if coords.shape[1] < n_components:
        pad = np.zeros((n, n_components - coords.shape[1]))
        coords = np.hstack([coords, pad])
    return coords.astype(np.float32)

def load_alignment_and_tree(fa_path: str, nwk_path: str):
    """
    Parses FASTA alignment and Newick tree into PyTorch-ready input tensors.
    """
    tree_obj = Phylo.read(nwk_path, 'newick')
    taxa = [term.name for term in tree_obj.get_terminals() if term.name]
    dist_mat = compute_fast_dist_matrix(tree_obj, taxa)
    mds_coords = compute_mds_coordinates(dist_mat, n_components=4)
    
    seq_dict = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(fa_path, 'fasta')}
    n_taxa = len(taxa)
    first_seq = list(seq_dict.values())[0]
    L = len(first_seq) // 3
    
    c_all = np.zeros((L, n_taxa, 1), dtype=np.int64)
    a_all = np.zeros((L, n_taxa, 1), dtype=np.int64)
    for i, sp in enumerate(taxa):
        seq = seq_dict.get(sp, '-' * (L * 3))
        for site in range(L):
            codon = seq[site*3 : (site+1)*3].upper()
            c_all[site, i, 0] = get_codon_token(codon)
            a_all[site, i, 0] = get_aa_token(codon)
            
    is_aa_invariable = np.zeros(L, dtype=bool)
    for site in range(L):
        aa_col = a_all[site, :, 0]
        valid_aa = aa_col[aa_col < 20]
        if len(np.unique(valid_aa)) <= 1:
            is_aa_invariable[site] = True
            
    c_tensor = torch.tensor(c_all, dtype=torch.long)
    a_tensor = torch.tensor(a_all, dtype=torch.long)
    d_tensor = torch.tensor(dist_mat, dtype=torch.float32).unsqueeze(0).repeat(L, 1, 1)
    z_tensor = torch.tensor(mds_coords, dtype=torch.float32).unsqueeze(0).repeat(L, 1, 1)
    
    return c_tensor, a_tensor, d_tensor, z_tensor, is_aa_invariable, taxa, L
