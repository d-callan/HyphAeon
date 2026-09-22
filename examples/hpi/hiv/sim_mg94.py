"""MG94 codon simulator with branch-specific omega (episodic selection).

Simulates codon evolution along a fixed tree. The internal branch leading to
the foreground clade (t0-t9) gets elevated omega -> those taxa share the
nonsynonymous substitutions from that selected branch (shared/episodic
selection = the selection-correlation signal we want to detect).

Writes a codon FASTA + the tree. Run twice (different seed) for host + par.
"""
import sys
import numpy as np
from scipy.linalg import expm

# --- genetic code (universal) ---
BASES = "TCAG"
AA = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
CODONS = [a + b + c for a in BASES for b in BASES for c in BASES]
SENSE = [c for c, a in zip(CODONS, AA) if a != "*"]          # 61 sense codons
AA_OF = {c: a for c, a in zip(CODONS, AA)}
IDX = {c: i for i, c in enumerate(SENSE)}
N = len(SENSE)


def q_matrix(omega, alpha=1.0, freqs=None):
    """MG94 rate matrix over the 61 sense codons.
    Single-nucleotide changes only; syn rate=alpha, nonsyn rate=alpha*omega.
    Rows sum to 0. freqs = equilibrium codon frequencies (target codon)."""
    if freqs is None:
        freqs = np.full(N, 1.0 / N)
    Q = np.zeros((N, N))
    for i, ci in enumerate(SENSE):
        for j, cj in enumerate(SENSE):
            if i == j:
                continue
            diff = [k for k in range(3) if ci[k] != cj[k]]
            if len(diff) != 1:
                continue                                   # only single-nt
            syn = AA_OF[ci] == AA_OF[cj]
            Q[i, j] = (alpha if syn else alpha * omega) * freqs[j]
    for i in range(N):
        Q[i, i] = -Q[i].sum()
    # normalize so mean rate = 1 substitution/unit (branch len = expected subs)
    pi = freqs
    scale = -(pi * Q.diagonal()).sum()
    return Q / scale


def simulate(n_sites, seed, fg_taxa=None, n_taxa=40, fg_omega=5.0,
             bg_omega=0.15, fg_internal_len=0.5, tip_len=0.08):
    """Tree: root -> FGnode (len fg_internal_len, omega fg_omega) -> fg_taxa tips
             root -> each background taxon (len tip_len, omega bg_omega)  [star]

    fg_taxa = the selected clade (shared internal branch at fg_omega -> those
    taxa share the nonsyn substitutions from that branch). Background taxa are
    near-independent (star), so only the FG clade carries a shared selected
    signal. For a NEGATIVE control, give host and par different fg_taxa.
    """
    if fg_taxa is None:
        fg_taxa = [f"t{i}" for i in range(10)]
    fg_set = set(fg_taxa)
    all_taxa = [f"t{i}" for i in range(n_taxa)]
    bg_taxa = [t for t in all_taxa if t not in fg_set]

    rng = np.random.default_rng(seed)
    freqs = np.full(N, 1.0 / N)
    P_fg_int = expm(q_matrix(fg_omega, freqs=freqs) * fg_internal_len)
    P_fg_tip = expm(q_matrix(fg_omega, freqs=freqs) * tip_len)
    P_bg_tip = expm(q_matrix(bg_omega, freqs=freqs) * tip_len)

    def step(P, parent):
        return rng.choice(N, p=P[parent])

    seqs = {t: [] for t in all_taxa}
    for _ in range(n_sites):
        root = rng.choice(N, p=freqs)
        fg_node = step(P_fg_int, root)
        for t in fg_taxa:
            seqs[t].append(SENSE[step(P_fg_tip, fg_node)])
        for t in bg_taxa:
            seqs[t].append(SENSE[step(P_bg_tip, root)])
    return {k: "".join(v) for k, v in seqs.items()}


def simulate_scatter(n_sites, seed, fg_taxa=None, n_taxa=40, n_sel=120,
                     bg_omega=0.15, tip_len=0.08):
    """Diffuse/convergent positive control for the BURDEN detector.

    fg_taxa are SCATTERED tips (NOT a clade — no shared internal branch). At
    n_sel 'selected' sites each fg taxon independently draws a derived
    (non-consensus) codon while background stays near the root -> high-LRT
    sites whose drivers are scattered individuals. At all other sites every
    taxon evolves under bg_omega from a shared root, so fg taxa are NOT
    phylogenetically divergent and do NOT cluster in the MDS ancestry coords.
    The coupled-burden signal is therefore per-individual, not clade/ancestry,
    and should survive ancestry correction — unlike the clade control, whose
    signal IS ancestry. Run host & par with the SAME scattered fg_taxa
    (different seeds) -> the same individuals are divergent in both genes.
    """
    if fg_taxa is None:
        # 12 scattered (non-contiguous) tips — deliberately not a clade
        fg_taxa = [f"t{i}" for i in (0, 4, 7, 11, 14, 18, 21, 25, 28, 31, 35, 38)]
    fg_set = set(fg_taxa)
    all_taxa = [f"t{i}" for i in range(n_taxa)]

    rng = np.random.default_rng(seed)
    freqs = np.full(N, 1.0 / N)
    P_bg = expm(q_matrix(bg_omega, freqs=freqs) * tip_len)

    def step(P, parent):
        return rng.choice(N, p=P[parent])

    sel_sites = set(rng.choice(n_sites, size=min(n_sel, n_sites), replace=False))
    seqs = {t: [] for t in all_taxa}
    for s in range(n_sites):
        root = rng.choice(N, p=freqs)
        for t in all_taxa:
            if t in fg_set and s in sel_sites:
                # derived: draw a codon whose AA differs from the root's AA
                root_aa = AA_OF[SENSE[root]]
                choices = [j for j in range(N) if AA_OF[SENSE[j]] != root_aa]
                seqs[t].append(SENSE[rng.choice(choices)])
            else:
                seqs[t].append(SENSE[step(P_bg, root)])
    return {k: "".join(v) for k, v in seqs.items()}


if __name__ == "__main__":
    # usage: sim_mg94.py out.fasta n_sites seed [fg_start fg_end] [--scatter]
    argv = [a for a in sys.argv[1:] if a != "--scatter"]
    scatter = "--scatter" in sys.argv
    out = argv[0] if len(argv) > 0 else "sim_out.fasta"
    n_sites = int(argv[1]) if len(argv) > 1 else 200
    seed = int(argv[2]) if len(argv) > 2 else 0
    fg = None
    if len(argv) > 4:
        fg = [f"t{i}" for i in range(int(argv[3]), int(argv[4]))]
    seqs = (simulate_scatter(n_sites, seed, fg_taxa=fg) if scatter
            else simulate(n_sites, seed, fg_taxa=fg))
    with open(out, "w") as f:
        for name, s in seqs.items():
            f.write(f">{name}\n{s}\n")
    print(f"wrote {out} ({len(seqs)} taxa x {n_sites} codons, fg={fg})")
