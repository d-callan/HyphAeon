#!/usr/bin/env python3
"""Generate the TN93-port oracle fixtures from the tn93 binary (v1.1.0).

Writes, next to this script:
  ambig_toy.fasta               hand-built pairs exercising ambiguity handling
  ambig_toy_<mode>_raw.csv      `tn93 -t 1000 -a <mode>` (raw values; 1000 = saturated)
                                for mode in RMYSKB, resolve, average, skip, gapmm
  ambig_toy_RMYSKB.csv          `tn93 -t 1.0 -a RMYSKB` (the reported pair set)
  dc80.fasta                    first 80 sequences of the Washington, DC alignment
                                (Perez-Losada 2017; validation/results/
                                Perez-Losada-2017-28961263/msa.fasta), with
                                columns that are gaps in all 80 removed (they cannot
                                change any pairwise TN93 value; this keeps the
                                fixture at ~150 kB)
  dc80_RMYSKB.csv, dc80_resolve.csv   `tn93 -t 1.0 -a <mode>` on dc80.fasta

All tn93 runs use the default minimum overlap (-l 100). tn93 is multithreaded
and writes pairs in a run-dependent order, so the data rows of every CSV are
sorted (header first); reruns are byte-identical.
Run: python3 gen_tn93_port_fixtures.py [path/to/dc/msa.fasta]
"""
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE = <trace5>/code/crates/trace50-core/tests/fixtures/tn93
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 6))
DC_MSA = (sys.argv[1] if len(sys.argv) > 1 else
          os.path.join(PROJECT_ROOT, "validation", "results", "Perez-Losada-2017-28961263", "msa.fasta"))


def tn93(fasta, out, threshold, mode):
    # tn93 prints its run summary JSON on stdout; the pair list goes to -o
    subprocess.run(["tn93", "-q", "-t", threshold, "-a", mode, "-o", out, fasta],
                   check=True, stdout=subprocess.DEVNULL)
    with open(out) as fh:
        lines = fh.read().splitlines()
    with open(out, "w") as fh:
        fh.write("\n".join([lines[0]] + sorted(lines[1:])) + "\n")


def write_fasta(path, records):
    with open(path, "w") as fh:
        for name, s in records:
            fh.write(f">{name}\n{s}\n")


# ---------------------------------------------------------------- toy cases
W = 240
rng = random.Random(6)
base = [rng.choice("ACGT") for _ in range(W)]
TS = {"A": "G", "G": "A", "C": "T", "T": "C"}


def edit(changes, src=None):
    s = list(base if src is None else src)
    for p, c in changes.items():
        s[p] = TS[s[p]] if c == "ts" else c
    return "".join(s)


def where(base_char, k, start=0):
    """first k positions >= start where base has base_char"""
    return [i for i in range(start, W) if base[i] == base_char][:k]


toy = [("base", "".join(base))]
# W (A/T, averaged under RMYSKB) opposite A; plus two transitions
toy.append(("W_on_A", edit({**{p: "W" for p in where("A", 6)}, 100: "ts", 150: "ts"})))
# R (A/G, resolved under RMYSKB) opposite A
toy.append(("R_on_A", edit({**{p: "R" for p in where("A", 6)}, 120: "ts"})))
# M (A/C) and K (G/T) at the same columns: ambiguity vs ambiguity, no common base
mk_cols = where("C", 5, 30)
toy.append(("M_cols", edit({**{p: "M" for p in mk_cols}, 60: "ts"})))
toy.append(("K_cols", edit({**{p: "K" for p in mk_cols}, 200: "ts"})))
# N, ?, B/D/H/V, U and lower case
toy.append(("N_etc", edit({5: "N", 6: "N", 7: "?", 8: "B", 9: "D", 10: "H", 11: "V", 90: "ts"})))
toy.append(("U_lower", edit({p: ("U" if base[p] == "T" else base[p].lower()) for p in range(0, W, 3)})))
# internal and terminal gap runs
toy.append(("gappy", edit({**{p: "-" for p in range(40, 70)}, **{p: "-" for p in range(225, 240)}, 180: "ts"})))
# no T at all in either sequence -> a zero base frequency -> K2P branch
noT = ["ACG"[i % 3] for i in range(W)]
toy.append(("noT_1", "".join(noT)))
toy.append(("noT_2", edit({p: "ts" for p in (3, 50, 77)}, noT)))
# every site a transition of the other -> saturated (tn93 returns 1000)
toy.append(("sat_1", "".join("AC"[i % 2] for i in range(W))))
toy.append(("sat_2", "".join("GT"[i % 2] for i in range(W))))
# two unrelated random sequences (high divergence)
toy.append(("rand_1", "".join(rng.choice("ACGT") for _ in range(W))))
toy.append(("rand_2", "".join(rng.choice("ACGT") for _ in range(W))))

toy_fa = os.path.join(HERE, "ambig_toy.fasta")
write_fasta(toy_fa, toy)
for mode in ("RMYSKB", "resolve", "average", "skip", "gapmm"):
    tn93(toy_fa, os.path.join(HERE, f"ambig_toy_{mode}_raw.csv"), "1000", mode)
tn93(toy_fa, os.path.join(HERE, "ambig_toy_RMYSKB.csv"), "1.0", "RMYSKB")

# ---------------------------------------------------------------- DC subset
recs = []
with open(DC_MSA) as fh:
    for line in fh:
        line = line.strip()
        if line.startswith(">"):
            if len(recs) == 80:
                break
            recs.append([line[1:].split()[0], []])
        elif line:
            recs[-1][1].append(line)
recs = [(n, "".join(p).upper()) for n, p in recs]
width = len(recs[0][1])
keep = [i for i in range(width) if any(s[i] != "-" for _, s in recs)]
dc80 = [(n, "".join(s[i] for i in keep)) for n, s in recs]
dc_fa = os.path.join(HERE, "dc80.fasta")
write_fasta(dc_fa, dc80)
for mode in ("RMYSKB", "resolve"):
    tn93(dc_fa, os.path.join(HERE, f"dc80_{mode}.csv"), "1.0", mode)
print(f"dc80: {len(dc80)} sequences, {len(keep)} of {width} columns kept")
