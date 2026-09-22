#!/usr/bin/env python3
"""SELECTION-correlation positive control (distinct from inject_coselect.py's
ALLELE correlation). Plants the SAME phylogenetic clade under selection at a
host site and a parasite site — the signature the model is actually designed
to detect (correlated selection-driving lineages, not correlated alleles).

Approach: find a tight clade in the parasite alignment (p-distance -> a cluster
of related individuals), then give that clade a derived (non-consensus) amino
acid at one host site AND one parasite site. A clade-clustered nonsynonymous
change is an episodic-selection signature — if the model registers selection at
both sites driven by the same lineage, their selection-driver profiles correlate
-> a coselection edge. This is the correct control for the selection-correlation
notion (vs inject_coselect.py's allele correlation).
"""
import sys
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq


def read(path):
    d = {}
    for r in SeqIO.parse(path, "fasta"):
        sid = r.id.split("|")[0]
        if sid not in d:
            d[sid] = str(r.seq).upper()
    return d


def aa(codon):
    try:
        return str(Seq(codon).translate())
    except Exception:
        return "X"


def pdist(seqs):
    cm = {}
    M = np.array([[cm.setdefault(c, len(cm)) if c not in "-NXnx?" else -1
                   for c in s] for s in seqs])
    v = M >= 0
    n = len(M)
    D = np.zeros((n, n))
    for i in range(n):
        b = v[i] & v[i + 1:]
        comp = b.sum(1)
        diff = ((M[i] != M[i + 1:]) & b).sum(1)
        D[i, i + 1:] = np.where(comp > 0, diff / np.maximum(comp, 1), 0)
        D[i + 1:, i] = D[i, i + 1:]
    return D


def main():
    host_fa, par_fa, out_host, out_par = sys.argv[1:5]
    clade_size = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    host, par = read(host_fa), read(par_fa)
    inds = sorted(set(host) & set(par))
    Lh = len(host[inds[0]]) // 3
    Lp = len(par[inds[0]]) // 3

    # find a clade: cluster par sequences by p-distance, take a tight cluster
    D = pdist([par[i] for i in inds])
    # greedy: pick the individual with most close neighbours, grow cluster
    order = np.argsort(D.sum(1))  # most central first
    clade = [order[0]]
    while len(clade) < clade_size:
        # add the individual closest to the current clade
        d_to_clade = D[:, clade].min(1)
        d_to_clade[clade] = 1e9
        clade.append(int(np.argmin(d_to_clade)))
    clade_inds = [inds[i] for i in clade]
    print(f"[*] clade of {len(clade_inds)} individuals "
          f"(mean within-dist {D[np.ix_(clade,clade)].mean():.3f})")

    # host site + par site to inject. RECURRENT nonsynonymous changes on the
    # clade (each clade member gets a different non-consensus AA) — this is the
    # diversifying/episodic-selection signature that registers a high LRT,
    # unlike a single clade-defining substitution (just a polymorphism).
    h_site = Lh // 3
    p_site = Lp // 3
    # a few distinct non-consensus codons to cycle through the clade
    host_codons = ["ACG", "AAG", "GAG", "CTG"]  # Thr Lys Glu Leu
    par_codons = ["AGG", "GAG", "TTG", "ACG"]   # Arg Glu Leu Thr
    host_new, par_new = {}, {}
    ci = 0
    for i in inds:
        s = host[i]
        if i in clade_inds:
            host_new[i] = s[:3*h_site] + host_codons[ci % len(host_codons)] + s[3*h_site+3:]
        else:
            host_new[i] = s
        sp = par[i]
        if i in clade_inds:
            par_new[i] = sp[:3*p_site] + par_codons[ci % len(par_codons)] + sp[3*p_site+3:]
        else:
            par_new[i] = sp
        ci += 1

    with open(out_host, "w") as f:
        for i in inds:
            f.write(f">{i}|gag|x\n{host_new[i]}\n")
    with open(out_par, "w") as f:
        for i in inds:
            f.write(f">{i}|prot|x\n{par_new[i]}\n")
    print(f"[*] PLANTED selection-correlation: host codon {h_site+1} + par codon "
          f"{p_site+1} both derived in the SAME {len(clade_inds)}-individual clade")
    print(f"    -> {out_host} , {out_par}")


if __name__ == "__main__":
    main()
