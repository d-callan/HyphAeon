#!/usr/bin/env python3
"""Positive control for coselection methods: inject a KNOWN host x parasite
correlation into the real Gag/Protease codon alignments, so we can measure
each method's power (does it recover the planted pair?) against the real
relatedness background — a baseline the real data lacks.

Injection: pick a variable parasite site, binarize it (major/minor codon),
then set a chosen host site's codon to one of two codons tracking the
parasite allele (with `strength` = fraction of individuals that follow the
rule; 1.0 = perfect correlation, 0 = none). Writes modified host fasta.
"""
import sys, random
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


def main():
    host_fa, par_fa, out_fa = sys.argv[1], sys.argv[2], sys.argv[3]
    strength = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    rng = random.Random(seed)

    host, par = read(host_fa), read(par_fa)
    inds = sorted(set(host) & set(par))
    Lh = len(host[inds[0]]) // 3
    Lp = len(par[inds[0]]) // 3

    # parasite site with AMINO-ACID-level variation (the model's delta is a
    # non-consensus-AA indicator, so a synonymous split like TTG/CTG=Leu/Leu
    # is invisible). Pick the site with the most balanced minor-AA fraction.
    def codon(seq, i): return seq[3 * i:3 * i + 3]
    from collections import Counter
    best_p, best_min = None, -1
    for j in range(Lp):
        aac = Counter(aa(codon(par[i], j)) for i in inds)
        aac = {k: v for k, v in aac.items() if k not in "X*"}
        if len(aac) < 2:
            continue
        maj_aa, mino_n = max(aac, key=aac.get), sorted(aac.values())[-2]
        frac = mino_n / len(inds)
        if 0.2 <= frac <= 0.8 and mino_n > best_min:
            best_p, best_min, best_maj = j, mino_n, maj_aa
    if best_p is None:
        sys.exit("no par site with AA-level variation")
    print(f"[*] planted par site = codon {best_p+1} (minor AA count={best_min}, "
          f"major AA={best_maj})")

    # host site to overwrite: inject two codons encoding DIFFERENT amino acids
    # (Ala vs Thr) tracking the par site's major/minor AA.
    h_site = Lh // 2
    A, B = "GCG", "ACG"  # Ala vs Thr
    new_host = {}
    n_follow = 0
    for i in inds:
        follow = rng.random() < strength
        par_minor = aa(codon(par[i], best_p)) != best_maj
        inj = (B if par_minor else A) if follow else (A if par_minor else B)
        s = host[i]
        new_host[i] = s[:3 * h_site] + inj + s[3 * h_site + 3:]
        n_follow += follow
    print(f"[*] injected host codon {h_site+1} tracks par {best_p+1} AA "
          f"(strength={strength}, {n_follow}/{len(inds)} follow)")

    with open(out_fa, "w") as f:
        for i in inds:
            f.write(f">{i}|gag|x\n{new_host[i]}\n")
    print(f"[*] wrote {out_fa} | PLANTED: host:{h_site+1} x par:{best_p+1}")


if __name__ == "__main__":
    main()
