"""Generate a multi-gene simulated panel for the `symbaeon screen` triage.

Builds a collection of host-gene and pathogen-gene codon MSAs that all share
the same individual axis (t0..t{n-1}), with a known ground truth:

  coupled-scatter  host & par share the same SCATTERED fg_taxa -> non-ancestry
                   burden coupling (the signal resid is built to catch)
  coupled-clade    host & par share the same CLADE fg_taxa -> high RAW burden
                   but the signal IS ancestry (resid should be low)
  active           selected (clade or scatter) but the fg_taxa are NOT shared
                   with any partner -> high activity, no coupling
  housekeeping     no foreground, purifying background only -> low activity

Usage: python sim_panel.py OUT_DIR [n_taxa] [n_sites]
Writes OUT_DIR/host/*.fasta, OUT_DIR/par/*.fasta, OUT_DIR/ground_truth.tsv.
"""
import os
import sys

import sim_mg94 as sim

# fg_taxa sets over the shared axis (defined for n_taxa=60; see below).
SCAT_A = [0, 5, 9, 13, 17, 22, 26, 30, 34, 38, 43, 47]
SCAT_B = [2, 7, 11, 16, 20, 25, 29, 33, 37, 42, 46, 50]
CLADE_A = list(range(0, 10))
ACT_H1 = list(range(10, 20))                       # clade unique to host
ACT_H2 = [1, 6, 12, 18, 24, 31, 36, 41, 48, 53, 55, 58]  # scatter unique to host
ACT_P1 = list(range(20, 30))                       # clade unique to par
ACT_P2 = [3, 8, 14, 19, 23, 28, 32, 39, 44, 49, 54, 57]  # scatter unique to par

# (gene, kind, fg_taxa, seed). kind: clade|scatter|hk
HOST = [
    ("h_scatA", "scatter", SCAT_A, 11),
    ("h_scatB", "scatter", SCAT_B, 12),
    ("h_cladeA", "clade", CLADE_A, 13),
    ("h_act1", "clade", ACT_H1, 14),
    ("h_act2", "scatter", ACT_H2, 15),
    ("h_hk1", "hk", None, 16),
    ("h_hk2", "hk", None, 17),
    ("h_hk3", "hk", None, 18),
]
PAR = [
    ("p_scatA", "scatter", SCAT_A, 21),
    ("p_scatB", "scatter", SCAT_B, 22),
    ("p_cladeA", "clade", CLADE_A, 23),
    ("p_act1", "clade", ACT_P1, 24),
    ("p_act2", "scatter", ACT_P2, 25),
    ("p_hk1", "hk", None, 26),
    ("p_hk2", "hk", None, 27),
    ("p_hk3", "hk", None, 28),
]
# ground-truth coupled pairs + the kind of coupling planted
COUPLED = [
    ("h_scatA", "p_scatA", "scatter"),
    ("h_scatB", "p_scatB", "scatter"),
    ("h_cladeA", "p_cladeA", "clade"),
]


def _fg(fg, n_taxa):
    return None if fg is None else [f"t{i}" for i in fg if i < n_taxa]


def _gen(kind, n_sites, seed, fg, n_taxa):
    if kind == "scatter":
        return sim.simulate_scatter(n_sites, seed, fg_taxa=_fg(fg, n_taxa),
                                    n_taxa=n_taxa)
    if kind == "clade":
        return sim.simulate(n_sites, seed, fg_taxa=_fg(fg, n_taxa),
                            n_taxa=n_taxa)
    # housekeeping: no foreground -> pure purifying background (star tree)
    return sim.simulate(n_sites, seed, fg_taxa=[], n_taxa=n_taxa,
                        bg_omega=0.2)


def _write(seqs, path):
    with open(path, "w") as f:
        for name, s in seqs.items():
            f.write(f">{name}\n{s}\n")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "panel"
    n_taxa = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    n_sites = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    hd, pd = os.path.join(out, "host"), os.path.join(out, "par")
    os.makedirs(hd, exist_ok=True)
    os.makedirs(pd, exist_ok=True)
    for name, kind, fg, seed in HOST:
        _write(_gen(kind, n_sites, seed, fg, n_taxa),
               os.path.join(hd, f"{name}.fasta"))
    for name, kind, fg, seed in PAR:
        _write(_gen(kind, n_sites, seed, fg, n_taxa),
               os.path.join(pd, f"{name}.fasta"))
    with open(os.path.join(out, "ground_truth.tsv"), "w") as f:
        f.write("host_gene\tpar_gene\tcoupling\n")
        for h, p, k in COUPLED:
            f.write(f"{h}\t{p}\t{k}\n")
    print(f"wrote {out}: {len(HOST)} host x {len(PAR)} par genes, "
          f"{n_taxa} taxa x {n_sites} codons; {len(COUPLED)} coupled pairs")


if __name__ == "__main__":
    main()
